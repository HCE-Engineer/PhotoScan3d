"""Synthetic photos under varied conditions: errors must stay small and be covered by the
reported uncertainty (acceptance from the spec: >= 90 % within ±0.3 mm)."""
import pytest

from synthetic_run import DEFAULT_PART, run

CASES = {
    "nominal": ({}, {}),
    "tilt12": ({}, {"tilt_deg": (12.0, -6.0)}),
    "close": ({}, {"dist_mm": 300.0, "f": 2100.0}),
    "thick10": ({**DEFAULT_PART, "thickness": 10.0}, {}),
    "rot55": ({**DEFAULT_PART, "place": (105.0, 60.0, 55.0)}, {"tilt_deg": (-6.0, 5.0)}),
    "thick10_tilt12": ({**DEFAULT_PART, "thickness": 10.0}, {"tilt_deg": (12.0, -6.0)}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_synthetic(case, tmp_path):
    part_kw, cam_kw = CASES[case]
    s, res, _ = run(tmp_path, part_kw or None, cam_kw or None, seed=list(CASES).index(case), verbose=False)
    assert s["n_vertices"] == 5 and s["n_holes"] == 3, s
    assert s["frac_within_0.3"] >= 0.9, s
    assert s["frac_within_2sigma"] >= 0.9, s
    assert s["max_err_mm"] < 0.3, s


def test_part_touching_window_is_flagged(tmp_path):
    s, _, _ = run(tmp_path, {**DEFAULT_PART, "place": (35.0, 60.0, 55.0)}, None, seed=9, verbose=False)
    assert any("boş alanının kenarına" in w for w in s["warnings"])


def test_uncalibrated_uncertainty_covers_error(tmp_path):
    """No calibration, no EXIF (focal guessed ~13 % off): larger errors, but reported honestly."""
    s, _, _ = run(tmp_path, {**DEFAULT_PART, "thickness": 10.0}, {"tilt_deg": (12.0, -6.0)}, seed=3,
                  verbose=False, calibrated=False)
    assert s["n_vertices"] == 5 and s["n_holes"] == 3 and s["frac_within_2sigma"] >= 0.9, s
    assert any("kalibrasyonu yok" in w for w in s["warnings"])


@pytest.mark.parametrize("tilt", [(4.0, -3.0), (12.0, -6.0)])
def test_laptop_screen_as_base(tmp_path, tilt):
    """14" laptop screen as pattern + backlight; pattern 0.8 mm under the glass."""
    part = {**DEFAULT_PART, "place": (100.0, 35.0, 10.0)}
    cam = {"target": (154.5, 82.4), "dist_mm": 300.0, "tilt_deg": tilt}
    s, _, _ = run(tmp_path, part, cam, seed=21, verbose=False, screen=True)
    assert s["n_vertices"] == 5 and s["n_holes"] == 3, s
    assert s["max_err_mm"] < 0.2 and s["frac_within_2sigma"] >= 0.9, s
