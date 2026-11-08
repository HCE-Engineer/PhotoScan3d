"""Round parts (bearing-like ring): outer diameter, bore, roundness and concentricity."""
import os

import numpy as np
import pytest

from synthetic_run import SCREEN14


@pytest.mark.parametrize("offset", [0.0, 0.30])
def test_bearing_ring_on_screen(tmp_path, monkeypatch, offset):
    monkeypatch.setenv("FOTO_OLCUM_PROJECTS", str(tmp_path))
    from foto_olcum import build, compare, features2d, synth
    from foto_olcum.board import screen_layout
    from foto_olcum.calib import Camera
    from foto_olcum.project import Measurement, Project

    sheet = screen_layout("synth", SCREEN14)
    sheet.measured_ref = [sheet.ref_nominal] * 2
    sheet.save(tmp_path / "_altliklar" / "synth" / "sheet.json")
    part = synth.Part(outer_dia=47.0, holes=[(offset, 0.0, 20.0)], thickness=14.0, place=(150.0, 80.0, 0.0))
    cam = synth.Cam(target=(154.5, 82.4), dist_mm=300.0, tilt_deg=(8.0, -5.0))
    photo = tmp_path / "ring.jpg"
    synth.render(part, sheet, cam, seed=5).save(photo, quality=92)
    Camera("synth", [cam.w, cam.h], cam.K().tolist(), [0] * 5, 0.0, 0, "charuco", 0.003).save()

    p = Project.create("ring")
    pid = p.add_photo(photo, "silhouette")
    p.update_photo(pid, sheet="synth", camera="synth")
    features2d.process_photo(p, pid)
    p.put_measurement(Measurement("thickness", 14.0, 0.02, "caliper", locked=True))
    r = features2d.extract(p, pid)
    o, h = r["outer"], r["holes"]
    assert o["shape"] == "circle" and len(h) == 1, r["outer"]
    assert abs(o["dia"] - 47.0) < 0.1 and abs(o["dia"] - 47.0) < 2 * o["sigma_dia"], o
    assert abs(h[0]["dia"] - 20.0) < 0.1 and abs(h[0]["dia"] - 20.0) < 2 * h[0]["sigma_dia"], h
    assert o["roundness_pv_mm"] < 0.1, o
    print(f"\nkaçıklık {offset}: dış {o['dia']:.3f}±{o['sigma_dia']:.3f} delik {h[0]['dia']:.3f}±{h[0]['sigma_dia']:.3f} dairesellik {o['roundness_pv_mm']:.3f} eşmerkezlilik {h[0]['concentricity_mm']:.3f}±{h[0]['sigma_concentricity_mm']:.3f}")
    ecc = h[0]["concentricity_mm"]
    assert abs(ecc - offset) < max(0.06, 2 * h[0]["sigma_concentricity_mm"]), h

    p.set_model(r["suggested_model"])
    b = build.build_project(p)
    assert b["ok"] and b["all_fully_constrained"], b
    c = compare.compare_project(p)
    assert c["summary"]["worst_max_dev_mm"] < 0.15, c["summary"]
