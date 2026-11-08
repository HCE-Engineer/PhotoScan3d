"""Synthetic end-to-end check of the photo pipeline. Usable from pytest and as a script.

Known part -> independent renderer -> process -> extract -> compare each measurement with
truth, expressed in the same part frame the pipeline chose.
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_PART = dict(
    outline=[(0, 0, 0), (80, 0, 6), (80, 45, 3), (30, 55, 0), (0, 55, 4)],
    holes=[(12, 12, 6.6), (65, 15, 9.0), (40, 35, 4.5)],
    thickness=5.0,
    place=(62.0, 75.0, 10.0),
)


SCREEN14 = {"connector": "eDP-1", "px": [1920, 1080], "mm": [309, 174]}


def prepare(root: Path, part_kw=None, cam_kw=None, seed=0, screen=False):
    """Sheet 'synth' (A4 paper, or the 14" laptop screen with glass), camera 'synth' and one
    rendered photo under root. Returns (photo, part)."""
    os.environ["FOTO_OLCUM_PROJECTS"] = str(root)
    from foto_olcum import synth
    from foto_olcum.board import Sheet
    from foto_olcum.calib import Camera

    part = synth.Part(**(part_kw or DEFAULT_PART))
    cam = synth.Cam(**(cam_kw or {}))
    if screen:
        from foto_olcum.board import screen_layout
        f = screen_layout("synth", SCREEN14)
        f.measured_ref = [f.ref_nominal, f.ref_nominal]
    else:
        f = Sheet("synth", measured_ref=[127.0, 127.0])
    f.save(root / "_altliklar" / "synth" / "sheet.json")
    photo = root / f"synth_{seed}.jpg"
    synth.render(part, f, cam, seed=seed).save(photo, quality=92)
    Camera("synth", [cam.w, cam.h], cam.K().tolist(), [0, 0, 0, 0, 0], 0.0, 0, "charuco", 0.0).save()
    return photo, part


def run(root: Path, part_kw=None, cam_kw=None, seed=0, verbose=True, calibrated=True, screen=False):
    from foto_olcum import features2d, synth
    from foto_olcum.project import Measurement, Project

    photo, part = prepare(root, part_kw, cam_kw, seed, screen)

    name = f"synth_{seed}"
    p = Project.create(name)
    pid = p.add_photo(photo, "silhouette")
    p.update_photo(pid, sheet="synth", camera="synth" if calibrated else None)
    proc = features2d.process_photo(p, pid)
    p.put_measurement(Measurement("thickness", part.thickness, 0.03, "caliper", "kalınlık", locked=True))
    res = features2d.extract(p, pid)

    # truth in the pipeline's part frame
    pf = res["part_frame"]
    o, ux = np.array(pf["origin_frame_mm"]), np.array(pf["x_dir"])
    uy = np.array([-ux[1], ux[0]])
    to_part = lambda q: np.array([(q - o) @ ux, (q - o) @ uy])
    tv = synth._place(part, [(x, y) for x, y, _ in part.outline])
    rows = []
    for v in res["outer"]["vertices"]:
        q = o + v["x"] * ux + v["y"] * uy
        k = int(np.argmin(np.linalg.norm(tv - q, axis=1)))
        tx, ty = to_part(tv[k])
        tr = part.outline[k][2]
        rows += [(f"{v['name']}.x", v["x"], tx, v["sigma_x"]), (f"{v['name']}.y", v["y"], ty, v["sigma_y"])]
        rows.append((f"{v['name']}.r", v["radius"], tr, max(v["sigma_r"], 1e-9)))
    th = [(synth._place(part, [(cx, cy)])[0], d) for cx, cy, d in part.holes]
    for h in res["holes"]:
        q = o + h["x"] * ux + h["y"] * uy
        k = int(np.argmin([np.linalg.norm(c - q) for c, _ in th]))
        tx, ty = to_part(th[k][0])
        rows += [(f"{h['name']}.x", h["x"], tx, h["sigma_xy"]), (f"{h['name']}.y", h["y"], ty, h["sigma_xy"]),
                 (f"{h['name']}.dia", h["dia"], th[k][1], h["sigma_dia"])]
    err = np.array([abs(a - b) for _, a, b, _ in rows])
    z = np.array([abs(a - b) / s for _, a, b, s in rows])
    summary = {"n": len(rows), "max_err_mm": float(err.max()), "rms_err_mm": float(np.sqrt(np.mean(err ** 2))),
               "frac_within_0.3": float(np.mean(err <= 0.3)), "frac_within_2sigma": float(np.mean(z <= 2)),
               "n_vertices": len(res["outer"]["vertices"]), "n_holes": len(res["holes"]),
               "pose_rms_px": proc["pose_rms_px"], "mm_per_px": proc["mm_per_px"],
               "warnings": proc["warnings"], "notes": res["notes"]}
    if verbose:
        for name_, a, b, s in rows:
            print(f"{name_:8s} ölçülen {a:9.3f}  gerçek {b:9.3f}  hata {a - b:+7.3f}  ±{s:.3f}  z={abs(a - b) / s:4.1f}")
        print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary, res, p


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="foto_synth_"))
    run(root)
    print("klasör:", root)
