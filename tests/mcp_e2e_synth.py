"""Full Claude-style flow over real MCP stdio on a synthetic photo, including FreeCAD.

photo -> process -> thickness (caliper) -> extract -> suggested model -> build -> compare,
then a deliberately wrong hole diameter must show up as a zone on that hole."""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_run import prepare  # noqa: E402


async def main(root: Path):
    photo, part = prepare(root)
    params = StdioServerParameters(command=os.path.join(sys.prefix, "bin", "foto-olcum-mcp"),
                                   env={**os.environ, "FOTO_OLCUM_PROJECTS": str(root)})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            async def call(tool, **kw):
                res = await s.call_tool(tool, kw)
                out = json.loads(res.content[0].text)
                n_img = sum(1 for c in res.content if c.type == "image")
                short = json.dumps(out, ensure_ascii=False)
                print(f"{tool} [{n_img} görsel] -> {short[:260]}")
                return out, n_img

            await call("new_project", name="e2e")
            out, _ = await call("add_photos", project="e2e", paths=[str(photo)], frame="synth", camera="synth")
            assert out["photos"][0]["markers_found"] >= 18, out
            out, n = await call("process_photos", project="e2e")
            assert out["ok"] and n == 1, out
            out, _ = await call("extract_features_2d", project="e2e", photo_id="P01")
            assert not out["ok"] and "thickness" in out["error"]
            await call("set_measurement", project="e2e", id="thickness", value_mm=part.thickness, label="kalınlık")
            ext, n = await call("extract_features_2d", project="e2e", photo_id="P01")
            assert ext["ok"] and n == 1 and len(ext["holes"]) == 3 and len(ext["outer"]["vertices"]) == 5
            model = ext["suggested_model"]
            out, _ = await call("set_model", project="e2e", model=model)
            assert out["ok"], out
            b, _ = await call("build_freecad", project="e2e")
            assert b["ok"] and b["all_fully_constrained"], b
            cmp_, n = await call("render_and_compare", project="e2e")
            print(json.dumps(cmp_["summary"]), json.dumps(cmp_["photos"][0]["zones"][:3], ensure_ascii=False))
            assert cmp_["summary"]["worst_max_dev_mm"] < 0.15, cmp_
            for a, bb in (("V1", "V2"), ("H1", None), ("L3", "H1"), ("H1", "H2")):
                m, _ = await call("measure", project="e2e", photo_id="P01", a=a, b=bb)
                assert m["ok"], m
            # deliberately wrong hole: zone must point at that hole
            h2 = next(x for x in (await call("status", project="e2e"))[0]["measurements"] if x["id"] == "h2_dia")
            await call("set_measurement", project="e2e", id="h2_dia", value_mm=h2["value_mm"] + 0.6,
                       source="photo_coarse", uncertainty_mm=0.2)
            await call("build_freecad", project="e2e")
            cmp2, _ = await call("render_and_compare", project="e2e")
            z = cmp2["photos"][0]["zones"][0]
            print("en büyük bölge:", z)
            assert z["nearest_feature"] == "hole_2" and z["max_dev_mm"] < -0.2, z
    print("E2E OK", root)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="foto_e2e_"))
    asyncio.run(main(root))
