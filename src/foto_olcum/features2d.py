"""Per-photo pipeline: pose, silhouette, subpixel edges, footprint, fits, part frame.

Every reported number carries an uncertainty built from:
  fit (statistical) ⊕ edge offset (unknown material bias B) ⊕ pose/sheet scale ⊕ thickness.
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np

from . import edges, fit, geometry, segment
from .calib import Pose, estimate_pose
from .board import Sheet
from .imaging import bbox_crop, put_label, save_debug, undistorted
from .project import Measurement, Project

EDGE_BIAS_PX = 0.5          # unknown material-dependent edge offset, 1 sigma, per edge


def _photo_paths(p: Project, pid: str):
    info = p.photos().get(pid)
    if not info:
        raise KeyError(f"foto bulunamadı: {pid}")
    return info, p.path(info["file"])


def process_photo(p: Project, pid: str) -> dict:
    info, raw = _photo_paths(p, pid)
    sheet = Sheet.load(info["sheet"])
    img, K, cam = undistorted(raw, info.get("camera"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    und_path = p.path(f"photos/undistorted/{pid}.png")
    und_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(und_path), img)
    pose = estimate_pose(gray, K, sheet)
    warnings = []
    if cam.source != "charuco":
        warnings.append("kamera kalibrasyonu yok: odak EXIF'ten tahmin edildi, distorsiyon sıfır "
                        "varsayıldı. Kenarlarda ±0.3 mm'ye kadar ek hata olabilir; calibrate_camera önerilir.")
    if pose.rms_px > 1.0:
        warnings.append(f"marker reprojeksiyon hatası yüksek ({pose.rms_px:.2f} px)")
    window_px = geometry.project(np.c_[sheet.window_poly(shrink=2.0), np.zeros(4)], K, pose.rvec, pose.tvec)
    roi = segment.roi_mask(gray.shape, window_px)
    sil = segment.backlit(gray, roi)
    warnings += sil.warnings
    cv2.imwrite(str(p.path(f"masks/{pid}.png")), sil.mask)
    tilt = float(np.degrees(np.arccos(abs(pose.R[2, 2]))))
    if tilt > 20:
        warnings.append(f"kamera dikten {tilt:.0f}° sapmış; 15°'nin altı önerilir")
    ms = p.measurements()
    t = ms["thickness"].value_mm if "thickness" in ms else 0.0
    center_px = geometry.project([[*np.mean(sheet.window_poly(), 0), t]], K, pose.rvec, pose.tvec)
    mmpp = float(geometry.mm_per_px(K, pose.rvec, pose.tvec, center_px, t)[0])

    dbg = img.copy()
    cv2.drawContours(dbg, [sil.outer.reshape(-1, 1, 2)], -1, (0, 200, 0), 2)
    for h in sil.holes:
        cv2.drawContours(dbg, [h.reshape(-1, 1, 2)], -1, (0, 140, 255), 2)
    cv2.polylines(dbg, [np.round(window_px).astype(np.int32)], True, (255, 0, 0), 2)
    out = save_debug(dbg, p.path(f"debug/{pid}_maske.png"), bbox_crop(window_px, gray.shape))
    p.update_photo(pid, pose=pose.to_json(), camera_source=cam.source, f_sigma_rel=cam.f_sigma_rel, mm_per_px=mmpp,
                   image_size=list(gray.shape[::-1]), processed=True)
    if not sheet.measured_ref:
        warnings.append("altlığın baskı ölçeği girilmedi (set_sheet_measurement): ölçülerde ±%1 ölçek belirsizliği var")
    return {"photo": pid, "pose_rms_px": round(pose.rms_px, 3), "charuco_corners": pose.n_corners,
            "camera_source": cam.source, "tilt_deg": round(tilt, 1),
            "camera_distance_mm": round(float(np.linalg.norm(pose.camera_center)), 0),
            "mm_per_px": round(mmpp, 4), "scale_sigma_rel": round(pose.scale_sigma_rel, 5),
            "holes_found": len(sil.holes), "contrast": [round(v) for v in sil.levels],
            "debug": out, "warnings": warnings}


@dataclass
class _Ctx:
    K: np.ndarray
    pose: Pose
    t: float
    t_sigma: float
    mmpp: float
    sheet: Sheet
    noise: float = 0.0
    extra_rel: float = 0.0      # intrinsics: focal error via height correction, lens distortion
    f_rel: float = 0.0          # relative focal-length uncertainty


def _contour_mm(gray, contour, c: _Ctx, want_ccw: bool):
    e = edges.refine(gray, contour)
    if len(e.pts) < 20:
        return None
    pts, n_mm, amb, _ = geometry.footprint(e.pts, e.normals, c.K, c.pose.rvec, c.pose.tvec, c.t)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if (area > 0) != want_ccw:
        pts, amb, img_pts = pts[::-1], amb[::-1], e.pts[::-1]
    else:
        img_pts = e.pts
    return pts, amb, img_pts


def _sys_sigma(c: _Ctx, length_mm: float, n_edges: int, amb: float = 0.0) -> float:
    """Systematic part: edge bias, scale (pose + sheet print), thickness ambiguity."""
    b = EDGE_BIAS_PX * c.mmpp * n_edges
    rel = np.sqrt(c.pose.scale_sigma_rel ** 2 + c.sheet.scale_sigma_rel ** 2 + c.extra_rel ** 2)
    s = abs(length_mm) * rel
    # top/bottom edge choice: its separation `amb` scales with thickness and focal errors
    pt = amb * np.hypot(c.t_sigma / c.t, c.f_rel) if c.t > 0 else amb
    return float(np.sqrt(b * b + s * s + pt * pt))


def extract(p: Project, pid: str, axis_line: int | None = None, origin_corner: int | None = None,
            create_measurements: bool = True, material: str = "") -> dict:
    info, _ = _photo_paths(p, pid)
    if not info.get("processed"):
        raise ValueError(f"{pid} işlenmemiş, önce process_photos")
    ms = p.measurements()
    if "thickness" not in ms:
        raise ValueError("parça kalınlığı gerekli: kumpasla ölçüp set_measurement(id='thickness', "
                         "value_mm=..., source='caliper') girin. Perspektif düzeltmesi buna bağlı.")
    sheet = Sheet.load(info["sheet"])
    pose = Pose.from_json(info["pose"])
    img = cv2.imread(str(p.path(f"photos/undistorted/{pid}.png")))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(str(p.path(f"masks/{pid}.png")), 0)
    c = _Ctx(pose.K, pose, ms["thickness"].value_mm, ms["thickness"].uncertainty_mm, info["mm_per_px"], sheet)
    # a focal-length error mis-scales the z = t plane relative to the sheet plane by
    # ~ t / D * df/f; an uncertain screen glass depth does the same by ~ dg / D;
    # uncalibrated lenses also carry unmodelled distortion
    dist_mm = float(np.linalg.norm(pose.camera_center))
    f_rel = c.f_rel = info.get("f_sigma_rel", 0.3)
    c.extra_rel = float(np.sqrt((c.t / dist_mm * f_rel) ** 2 + (sheet.glass_sigma_mm / dist_mm) ** 2
                                + (0.002 if info.get("camera_source") != "charuco" else 0.0) ** 2))

    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    oi = max((i for i in range(len(contours)) if hier[i][3] < 0), key=lambda i: cv2.contourArea(contours[i]))
    outer = _contour_mm(gray, contours[oi].reshape(-1, 2), c, want_ccw=True)
    if outer is None:
        raise ValueError("dış kontur çıkarılamadı")
    opts, oamb, oimg = outer
    ds = max(c.mmpp, 0.05)
    lines, corners, samples = fit.segment(opts, ds, noise_mm=0.3 * c.mmpp, sharp_r=max(0.3, 3 * c.mmpp))
    if len(lines) < 2:
        raise ValueError("dış konturda düz kenar bulunamadı (dönel/serbest form parça şimdilik desteklenmiyor)")
    c.noise = float(np.median([ln.rms for ln in lines]))

    # part frame: x along the chosen (default: longest) line, y into the material (CCW contour)
    ai = int(np.argmax([ln.length for ln in lines])) if axis_line is None else axis_line - 1
    if not 0 <= ai < len(lines):
        raise ValueError(f"axis_line 1..{len(lines)} olmalı")
    ax = lines[ai]
    oc = (ai - 1) % len(lines) if origin_corner is None else origin_corner - 1
    if not 0 <= oc < len(corners):
        raise ValueError(f"origin_corner 1..{len(corners)} olmalı")
    origin = corners[oc].p
    ux = ax.u
    uy = np.array([-ux[1], ux[0]])

    def to_part(q):
        d = np.asarray(q) - origin
        return np.array([d @ ux, d @ uy])

    order = [(oc + k) % len(corners) for k in range(len(corners))]
    amb_med = float(np.median(oamb))
    verts = []
    for k, ci in enumerate(order, start=1):
        cr = corners[ci]
        xy = to_part(cr.p)
        rot = abs(xy[0]) * ax.sigma_angle        # axis direction uncertainty
        stat = np.hypot(cr.sigma_p if np.isfinite(cr.sigma_p) else 0.5, corners[oc].sigma_p)
        amb_v = max(amb_med, float(oamb[np.argmin(np.linalg.norm(opts - cr.p, axis=1))]))
        sx = float(np.sqrt(stat ** 2 + _sys_sigma(c, xy[0], 2, amb_v) ** 2))
        sy = float(np.sqrt(stat ** 2 + rot ** 2 + _sys_sigma(c, xy[1], 2, amb_v) ** 2))
        verts.append({"name": f"V{k}", "corner": ci + 1, "x": float(xy[0]), "y": float(xy[1]),
                      "sigma_x": sx, "sigma_y": sy, "radius": cr.radius,
                      "sigma_r": float(np.sqrt((cr.sigma_r if np.isfinite(cr.sigma_r) else 0.5) ** 2
                                               + _sys_sigma(c, cr.radius, 1, amb_v) ** 2)) if cr.radius > 0 else 0.0,
                      "turn_deg": round(cr.turn_deg, 2), "explained": bool(cr.explained),
                      "fit_rms_mm": None if not np.isfinite(cr.rms) else round(cr.rms, 4)})

    holes, pockets = [], []
    hole_idx = [i for i in range(len(contours)) if hier[i][3] == oi and cv2.contourArea(contours[i]) >= 30]
    for i in hole_idx:
        hc = _contour_mm(gray, contours[i].reshape(-1, 2), c, want_ccw=False)
        if hc is None:
            continue
        hpts, hamb, himg = hc
        circ = fit.fit_circle(hpts)
        # an uncertain focal length bends the top/bottom arcs apart by ~ amb * df/f
        circ_tol = max(4 * c.noise, 0.03, 0.15 * c.mmpp, float(np.median(hamb)) * c.f_rel)
        if circ.rms < circ_tol and circ.r > 0.5:
            xy = to_part(circ.c)
            sc = float(np.sqrt(circ.sigma_c ** 2 + _sys_sigma(c, np.hypot(*xy), 1, float(np.median(hamb))) ** 2
                               + (abs(xy[0]) * ax.sigma_angle) ** 2))
            sd = float(np.sqrt((2 * circ.sigma_r) ** 2 + _sys_sigma(c, 2 * circ.r, 2, float(np.median(hamb))) ** 2))
            holes.append({"x": float(xy[0]), "y": float(xy[1]), "dia": 2 * circ.r, "sigma_xy": sc,
                          "sigma_dia": sd, "fit_rms_mm": round(circ.rms, 4), "img": himg})
        else:
            hl, hcn, _ = fit.segment(hpts, ds, noise_mm=c.noise, sharp_r=max(0.3, 3 * c.mmpp))
            pockets.append({"n_lines": len(hl), "circle_rms_mm": round(circ.rms, 4),
                            "corners": [{"x": float(to_part(k.p)[0]), "y": float(to_part(k.p)[1]),
                                         "radius": k.radius, "explained": bool(k.explained)} for k in hcn],
                            "img": himg})
    holes.sort(key=lambda h: (round(h["y"], 0), h["x"]))
    for j, h in enumerate(holes, 1):
        h["name"] = f"H{j}"
    for j, pk in enumerate(pockets, 1):
        pk["name"] = f"K{j}"

    debug = _draw(img, c, oimg, lines, corners, order, holes, pockets, origin, ux, uy, samples, p, pid)
    unexplained = [v["name"] for v in verts if not v["explained"]]
    result = {
        "photo": pid, "thickness_mm": c.t, "mm_per_px": round(c.mmpp, 4),
        "edge_noise_mm": round(c.noise, 4),
        "part_frame": {"axis_line": ai + 1, "origin_corner": oc + 1,
                       "origin_frame_mm": origin.round(4).tolist(), "x_dir": ux.round(6).tolist()},
        "outer": {"n_lines": len(lines),
                  "lines": [{"name": f"L{i + 1}", "length_mm": round(ln.length, 3), "rms_mm": round(ln.rms, 4),
                             "point": to_part(ln.c).round(4).tolist(),
                             "dir": [round(float(ln.u @ ux), 6), round(float(ln.u @ uy), 6)],
                             "sigma_offset_mm": round(float(np.hypot(ln.sigma_offset, _sys_sigma(c, 0, 1))), 4),
                             "sigma_angle_rad": float(ln.sigma_angle)}
                            for i, ln in enumerate(lines)],
                  "vertices": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in vv.items()} for vv in verts]},
        "holes": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k != "img"} for h in holes],
        "pockets": [{k: v for k, v in pk.items() if k != "img"} for pk in pockets],
        "debug": debug,
        "notes": [],
    }
    if unexplained:
        result["notes"].append(f"{unexplained} köşeleri basit keskin köşe/teğet radyusla açıklanamadı "
                               "(çentik, pah veya eğri olabilir): debug görseline bakın")
    if pockets:
        result["notes"].append(f"{len(pockets)} dairesel olmayan iç boşluk var (K*): pocket olarak modelleyin")

    p.update_photo(pid, part_frame=result["part_frame"])
    (p.path(f"debug/{pid}_ozellikler.json")).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    if create_measurements:
        made, model = _measurements(p, pid, verts, holes, material, debug["preview"])
        result["measurements"] = made
        result["suggested_model"] = model
    return result


def _measurements(p, pid, verts, holes, material, evidence):
    items, prof = [], []
    for k, v in enumerate(verts):
        vid = f"o_v{k + 1}"
        if k == 0:
            xs, ys = 0, 0
        else:
            xs = f"{vid}_x"
            items.append(Measurement(xs, v["x"], v["sigma_x"], "photo_coarse", f"Dış kontur {v['name']} x",
                                     material=material, feature_ref="outer", evidence=[evidence]))
            if abs(v["y"]) < 1e-6:
                ys = 0        # on the x axis by construction (default origin: V2)
            else:
                ys = f"{vid}_y"
                items.append(Measurement(ys, v["y"], v["sigma_y"], "photo_coarse", f"Dış kontur {v['name']} y",
                                         material=material, feature_ref="outer", evidence=[evidence]))
        vert = [xs, ys]
        if v["radius"] > 0:
            rid = f"{vid}_r"
            items.append(Measurement(rid, v["radius"], v["sigma_r"], "photo_coarse", f"{v['name']} köşe radyusu",
                                     material=material, feature_ref="outer", evidence=[evidence]))
            vert.append(rid)
        prof.append(vert)
    feats = [{"id": "body", "type": "pad", "plane": "XY", "profile": prof, "length": "thickness"}]
    for j, h in enumerate(holes, 1):
        base = f"h{j}"
        for suffix, val, sig, lab in (("x", h["x"], h["sigma_xy"], "merkez x"), ("y", h["y"], h["sigma_xy"], "merkez y"),
                                      ("dia", h["dia"], h["sigma_dia"], "çap")):
            items.append(Measurement(f"{base}_{suffix}", val, sig, "photo_coarse", f"Delik {h['name']} {lab}",
                                     material=material, feature_ref=h["name"], evidence=[evidence]))
        feats.append({"id": f"hole_{j}", "type": "hole", "center": [f"{base}_x", f"{base}_y"],
                      "diameter": f"{base}_dia", "depth": "through"})
    stored = p.put_photo_measurements(items)
    return stored, {"units": "mm", "features": feats}


def _draw(img, c, oimg, lines, corners, order, holes, pockets, origin, ux, uy, samples, p, pid):
    dbg = img.copy()
    pts = np.round(oimg).astype(np.int32)
    cv2.polylines(dbg, [pts.reshape(-1, 1, 2)], True, (0, 180, 0), 1, cv2.LINE_AA)
    z = c.t

    def to_img(q):
        return geometry.project([[q[0], q[1], z]], c.K, c.pose.rvec, c.pose.tvec)[0]

    for i, ln in enumerate(lines):
        a, b = samples[ln.idx[0]], samples[min(ln.idx[1], len(samples)) - 1]
        pa, pb = to_img(a), to_img(b)
        cv2.line(dbg, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)), (255, 120, 0), 2, cv2.LINE_AA)
        put_label(dbg, f"L{i + 1}", (pa + pb) / 2 + 8, (180, 80, 0))
    for k, ci in enumerate(order, 1):
        cr = corners[ci]
        q = to_img(cr.p)
        col = (0, 0, 230) if cr.explained else (0, 0, 120)
        cv2.circle(dbg, tuple(np.round(q).astype(int)), 6, col, 2)
        lab = f"V{k}" + (f" R{cr.radius:.2f}" if cr.radius > 0 else "") + ("" if cr.explained else " ?")
        put_label(dbg, lab, q + [8, -8], col)
    for h in holes:
        cv2.polylines(dbg, [np.round(h["img"]).astype(np.int32).reshape(-1, 1, 2)], True, (0, 140, 255), 1, cv2.LINE_AA)
        put_label(dbg, f"{h['name']} D{h['dia']:.2f}", h["img"].mean(0), (0, 100, 220))
    for pk in pockets:
        cv2.polylines(dbg, [np.round(pk["img"]).astype(np.int32).reshape(-1, 1, 2)], True, (200, 0, 200), 2)
        put_label(dbg, pk["name"], pk["img"].mean(0), (200, 0, 200))
    o = to_img(origin)
    for d, col, name in ((ux, (0, 0, 255), "X"), (uy, (0, 160, 0), "Y")):
        tip = to_img(origin + d * 15)
        cv2.arrowedLine(dbg, tuple(np.round(o).astype(int)), tuple(np.round(tip).astype(int)), col, 2, tipLength=0.2)
        put_label(dbg, name, tip, col, 0.7)
    return save_debug(dbg, p.path(f"debug/{pid}_ozellikler.png"), bbox_crop(oimg, dbg.shape))
