"""End-to-end smoke test over real stdio: new_project → measurements → model → FreeCAD build."""
import asyncio, json, os, sys, tempfile
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    root = tempfile.mkdtemp(prefix="foto_olcum_smoke_")
    params = StdioServerParameters(command=os.path.join(sys.prefix, "bin", "foto-olcum-mcp"),
                                   env={**os.environ, "FOTO_OLCUM_PROJECTS": root})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print("tools:", tools)
            async def call(tool, **kw):
                res = await s.call_tool(tool, kw)
                out = json.loads(res.content[0].text)
                print(tool, "->", json.dumps(out, ensure_ascii=False)[:300])
                return out
            await call("new_project", name="smoke")
            bad_id = await call("set_measurement", project="smoke", id="h", value_mm=1, source="photo_coarse", uncertainty_mm=0.2)
            assert not bad_id["ok"]
            for mid, v in [("w", 50), ("plate_h", 30), ("thick", 4), ("hx", 10), ("hy", 15), ("hd", 6.4)]:
                await call("set_measurement", project="smoke", id=mid, value_mm=v,
                           source="photo_coarse", uncertainty_mm=0.2)
            await call("set_measurement", project="smoke", id="thick", value_mm=4.02)
            locked = await call("set_measurement", project="smoke", id="thick", value_mm=9, source="photo_coarse")
            assert not locked["ok"], "locked value must not change"
            bad = await call("set_model", project="smoke", model={"features": [{"id": "x", "type": "pad", "profile": [[0, 0], ["nope", 0], [0, 1]], "length": "thick"}]})
            assert not bad["ok"]
            ok = await call("set_model", project="smoke", model={"features": [
                {"id": "base", "type": "pad", "profile": [[0, 0], ["w", 0, 3], ["w", "plate_h", 3], [0, "plate_h"]], "length": "thick"},
                {"id": "hole_1", "type": "hole", "center": ["hx", "hy"], "diameter": "hd", "depth": "through"}]})
            assert ok["ok"], ok
            b = await call("build_freecad", project="smoke")
            assert b["ok"] and b["all_fully_constrained"], b
            assert os.path.getsize(b["step"]) > 1000
            st = await call("status", project="smoke")
            assert st["locked"] == ["thick"]
    print("SMOKE OK", root)

asyncio.run(main())
