"""Derived dimensions between extracted 2D features, with uncertainty.

Systematic terms (scale, edge bias) are correlated between features; combining the two
sigmas in quadrature is a deliberate, simple and slightly conservative choice."""
import json

import numpy as np

from .project import Project


def _load(p: Project, pid: str) -> dict:
    f = p.path(f"debug/{pid}_ozellikler.json")
    if not f.exists():
        raise FileNotFoundError(f"{pid} için özellik yok, önce extract_features_2d")
    r = json.loads(f.read_text(encoding="utf-8"))
    feats = {}
    for v in r["outer"]["vertices"]:
        feats[v["name"]] = ("V", np.array([v["x"], v["y"]]), np.hypot(v["sigma_x"], v["sigma_y"]) / np.sqrt(2), v)
    for h in r["holes"]:
        feats[h["name"]] = ("H", np.array([h["x"], h["y"]]), h["sigma_xy"], h)
    for ln in r["outer"]["lines"]:
        feats[ln["name"]] = ("L", np.array(ln["point"]), ln["sigma_offset_mm"], ln)
    return feats


def measure(p: Project, pid: str, a: str, b: str | None) -> dict:
    feats = _load(p, pid)
    for n in (a, b):
        if n and n not in feats:
            raise KeyError(f"'{n}' yok; mevcut: {sorted(feats)}")
    ka, pa, sa, ra = feats[a]
    if b is None:
        if ka == "H":
            return {"kind": "diameter", "value_mm": ra["dia"], "uncertainty_mm": ra["sigma_dia"]}
        if ka == "V":
            return {"kind": "radius", "value_mm": ra["radius"], "uncertainty_mm": ra["sigma_r"]}
        return {"kind": "length", "value_mm": ra["length_mm"], "uncertainty_mm": None,
                "note": "doğru uzunluğu köşe radyuslarına bağlıdır; köşe-köşe mesafesi için V-V kullanın"}
    kb, pb, sb, rb = feats[b]
    if ka == "L" and kb != "L":
        ka, pa, sa, ra, kb, pb, sb, rb = kb, pb, sb, rb, ka, pa, sa, ra
    if kb == "L" and ka != "L":
        u = np.array(rb["dir"])
        n = np.array([-u[1], u[0]])
        d = float((pa - pb) @ n)
        s_line = np.hypot(sb, abs((pa - pb) @ u) * rb["sigma_angle_rad"])
        return {"kind": "point_to_line", "value_mm": abs(d), "uncertainty_mm": float(np.hypot(sa, s_line))}
    if ka == "L" and kb == "L":
        ua, ub = np.array(ra["dir"]), np.array(rb["dir"])
        ang = float(np.degrees(np.arccos(np.clip(abs(ua @ ub), 0, 1))))
        n = np.array([-ua[1], ua[0]])
        d = float(abs((pb - pa) @ n))
        lever = abs((pb - pa) @ ua)
        s = float(np.sqrt(sa ** 2 + sb ** 2 + (lever * ra["sigma_angle_rad"]) ** 2))
        out = {"kind": "line_to_line", "value_mm": d, "uncertainty_mm": s, "angle_deg": ang,
               "angle_sigma_deg": float(np.degrees(np.hypot(ra["sigma_angle_rad"], rb["sigma_angle_rad"])))}
        if ang > 1.0:
            out["note"] = "kenarlar paralel değil; mesafe L1 üzerindeki noktadan ölçüldü"
        return out
    d = pb - pa
    return {"kind": "center_distance", "value_mm": float(np.linalg.norm(d)),
            "uncertainty_mm": float(np.hypot(sa, sb)), "dx_mm": float(d[0]), "dy_mm": float(d[1])}
