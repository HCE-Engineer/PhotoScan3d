import math

import numpy as np
import pytest

from foto_olcum import fit, geometry, model
from foto_olcum.project import LockedError, Measurement, Project


def test_circle_fit_recovers_center_radius():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi, 400)
    p = np.c_[3 + 2.5 * np.cos(a), -1 + 2.5 * np.sin(a)] + rng.normal(0, 0.01, (400, 2))
    c = fit.fit_circle(p)
    assert np.allclose(c.c, [3, -1], atol=0.005) and abs(c.r - 2.5) < 0.005
    assert 0.005 < c.rms < 0.015 and c.sigma_r < 0.005


def test_line_fit_direction_and_offset():
    t = np.linspace(0, 40, 300)
    p = np.c_[t, 0.5 * t + 2] + np.random.default_rng(2).normal(0, 0.01, (300, 2))
    ln = fit.fit_line(p)
    assert abs(abs(ln.u[1] / ln.u[0]) - 0.5) < 1e-3 and ln.rms < 0.015


def _square(r):
    from foto_olcum import synth
    part = synth.Part(outline=[(0, 0, r), (50, 0, r), (50, 30, r), (0, 30, r)])
    return synth.outline_points(part, 0.5)


@pytest.mark.parametrize("r", [0, 2.0, 5.0])
def test_segment_square_corners(r):
    pts = _square(r) + np.random.default_rng(3).normal(0, 0.01, (len(_square(r)), 2))
    lines, corners, _ = fit.segment(pts, 0.1, noise_mm=0.01)
    assert len(lines) == 4
    for cr in corners:
        assert abs(cr.radius - r) < 0.1
    vs = sorted((round(c.p[0]), round(c.p[1])) for c in corners)
    assert vs == [(0, 0), (0, 30), (50, 0), (50, 30)]


def test_footprint_picks_inner_edge():
    # camera straight above the origin, 300 mm up; a 10 mm thick wall at x = 50 facing +x
    K = np.array([[3000, 0, 1500], [0, 3000, 1000], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])
    C = np.array([0, 0, 300.0])
    rvec = __import__("cv2").Rodrigues(R)[0]
    tvec = -R @ C
    top = geometry.project([[50, 0, 10]], K, rvec, tvec)        # outer wall: top edge is visible
    pts, n, amb, used_top = geometry.footprint(top, np.array([[1.0, 0]]), K, rvec, tvec, 10.0)
    assert used_top[0] and abs(pts[0, 0] - 50) < 1e-6 and amb[0] > 1
    # hole wall at x = 50 whose material is at x > 50: the bottom edge bounds the silhouette
    bot = geometry.project([[50, 0, 0]], K, rvec, tvec)
    pts, n, amb, used_top = geometry.footprint(bot, np.array([[-1.0, 0]]), K, rvec, tvec, 10.0)
    assert not used_top[0] and abs(pts[0, 0] - 50) < 1e-6


def test_model_validation_messages():
    vals = {"w": 10, "plate_h": 5, "thickness": 2}
    good = {"features": [{"id": "b", "type": "pad", "profile": [[0, 0], ["w", 0, 1], ["w", "plate_h"]], "length": "thickness"}]}
    assert model.validate(good, vals) == []
    bad = {"features": [{"id": "b", "type": "pad", "profile": [[0, 0], ["zz", 0]], "length": "thickness"},
                        {"id": "f", "type": "fillet"}]}
    errs = model.validate(bad, vals)
    assert any("profile" in e for e in errs) and any("henüz desteklenmiyor" in e for e in errs)
    assert model.evaluate("w / 2 + sqrt(plate_h ** 2)", vals) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        model.evaluate("__import__('os')", vals)


def test_caliper_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("FOTO_OLCUM_PROJECTS", str(tmp_path))
    p = Project.create("lock")
    p.put_measurement(Measurement("plate_w", 50.0, 0.2, "photo_coarse"))
    p.put_measurement(Measurement("plate_w", 50.12, 0.03, "caliper", locked=True))
    with pytest.raises(LockedError):
        p.put_measurement(Measurement("plate_w", 49.0, 0.2, "photo_coarse"))
    r = p.put_photo_measurements([Measurement("plate_w", 49.0, 0.2, "photo_coarse")])
    assert r["skipped_locked"] == ["plate_w"] and p.measurements()["plate_w"].value_mm == 50.12
    with pytest.raises(ValueError):
        Measurement("mm", 1, 0.1, "caliper")
