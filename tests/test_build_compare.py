"""FreeCAD build + compare on the suggested model, with a non-default part origin."""
import pytest

from foto_olcum import build, compare, features2d
from synthetic_run import run


@pytest.mark.parametrize("origin_corner", [None, 3])
def test_suggested_model_matches_photo(tmp_path, origin_corner):
    _, _, p = run(tmp_path, seed=11, verbose=False)
    res = features2d.extract(p, "P01", origin_corner=origin_corner)
    v2 = res["outer"]["vertices"][1]
    if origin_corner is not None:
        assert abs(v2["y"]) > 1, "V2 must not lie on the axis for this origin"
        assert res["suggested_model"]["features"][0]["profile"][1][1] == "o_v2_y"
    p.set_model(res["suggested_model"])
    b = build.build_project(p)
    assert b["ok"] and b["all_fully_constrained"], b
    c = compare.compare_project(p)
    s = c["summary"]
    assert s["worst_max_dev_mm"] < 0.15 and s["min_iou"] > 0.995, c
