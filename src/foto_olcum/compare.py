"""Project the built mesh into each photo and compare its silhouette with the real one.

Signed deviation per model-silhouette point: + = model has material the photo does not
(red), - = model is missing material (blue). Distances use the photo's subpixel edges.
"""
import cv2
import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from . import edges, geometry
from .calib import Pose
from .imaging import bbox_crop, put_label, save_debug
from .model import evaluate
from .project import Project

SS = 4                       # supersampling of the model rasterisation
ZONE_MM = 0.15               # deviations above this form reported zones


def _feature_geometry(p: Project):
    """Evaluated 2D geometry of the model features (part coords) for zone attribution."""
    vals = {k: m.value_mm for k, m in p.measurements().items()}
    out = []
    for f in p.model()["features"]:
        if "profile" in f:
            pts = [(evaluate(v[0], vals), evaluate(v[1], vals)) for v in f["profile"]]
            for i in range(len(pts)):
                out.append((f"{f['id']} kenar {i + 1}-{(i + 1) % len(pts) + 1}", "seg", (pts[i], pts[(i + 1) % len(pts)])))
        if f["type"] == "hole":
            c = (evaluate(f["center"][0], vals), evaluate(f["center"][1], vals))
            out.append((f["id"], "circle", (c, evaluate(f["diameter"], vals) / 2)))
    return out


def _nearest_feature(q, feats):
    best, bd = None, np.inf
    for name, kind, g in feats:
        if kind == "circle":
            d = abs(np.hypot(q[0] - g[0][0], q[1] - g[0][1]) - g[1])
        else:
            a, b = np.array(g[0]), np.array(g[1])
            t = np.clip(np.dot(q - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
            d = np.linalg.norm(q - (a + t * (b - a)))
        if d < bd:
            best, bd = name, d
    return best, float(bd)


def compare_photo(p: Project, pid: str, mesh: trimesh.Trimesh, feats) -> dict:
    info = p.photos()[pid]
    pf = info.get("part_frame")
    if not pf:
        return {"photo": pid, "skipped": "parça çerçevesi yok (önce extract_features_2d)"}
    pose = Pose.from_json(info["pose"])
    img = cv2.imread(str(p.path(f"photos/undistorted/{pid}.png")))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    photo_mask = cv2.imread(str(p.path(f"masks/{pid}.png")), 0)
    o, ux = np.array(pf["origin_frame_mm"]), np.array(pf["x_dir"])
    uy = np.array([-ux[1], ux[0]])
    v = mesh.vertices
    world = np.c_[o[0] + v[:, 0] * ux[0] + v[:, 1] * uy[0], o[1] + v[:, 0] * ux[1] + v[:, 1] * uy[1], v[:, 2]]
    uv = geometry.project(world, pose.K, pose.rvec, pose.tvec)

    x0, y0, x1, y1 = bbox_crop(uv, gray.shape, pad=40)
    big = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS), np.uint8)
    tri = np.round(((uv[mesh.faces] - [x0, y0]) * SS + (SS - 1) / 2) * 16).astype(np.int32)
    # one call per triangle: fillPoly on many overlapping polygons applies an even-odd rule
    for t3 in tri:
        cv2.fillConvexPoly(big, t3, 255, lineType=cv2.LINE_8, shift=4)
    model_crop = cv2.resize(big, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    model_mask = np.zeros_like(photo_mask)
    model_mask[y0:y1, x0:x1] = (model_crop >= 128) * 255

    inter = np.count_nonzero(model_mask & photo_mask)
    union = np.count_nonzero(model_mask | photo_mask)
    iou = inter / union if union else 0.0

    # model silhouette boundary (subpixel via supersampled contours)
    cs, _ = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    model_contours = [(c.reshape(-1, 2).astype(float) - (SS - 1) / 2) / SS + [x0, y0] for c in cs]
    mpts = np.concatenate(model_contours)
    # photo subpixel edges
    pcs, _ = cv2.findContours(photo_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    epts = np.concatenate([edges.refine(gray, c.reshape(-1, 2)).pts for c in pcs if len(c) > 20])
    d, _ = cKDTree(epts).query(mpts)
    inside_photo = map_coordinates(photo_mask.astype(np.float32), [mpts[:, 1], mpts[:, 0]], order=1) >= 128
    sign = np.where(inside_photo, -1.0, 1.0)     # boundary point inside the photo part: model too small
    mmpp = geometry.mm_per_px(pose.K, pose.rvec, pose.tvec, mpts, float(np.median(v[:, 2].max())))
    dev = sign * d * mmpp

    # zones: contiguous runs along the (subsampled) boundary
    step = max(1, len(mpts) // 4000)
    idx = np.arange(0, len(mpts), step)
    big_dev = np.abs(dev[idx]) > ZONE_MM
    t = float(v[:, 2].max())
    part_xy = lambda q: (lambda w: np.array([(w - o) @ ux, (w - o) @ uy]))(geometry.backproject(q[None], pose.K, pose.rvec, pose.tvec, t)[0])
    zones = []
    if big_dev.any():
        lab = np.cumsum(np.r_[1, np.diff(big_dev.astype(int)) != 0])
        for L in np.unique(lab[big_dev]):
            sel = idx[(lab == L) & big_dev]
            if len(sel) * step < 5:
                continue
            k = sel[np.argmax(np.abs(dev[sel]))]
            q = part_xy(mpts[k])
            name, dist = _nearest_feature(q, feats)
            zones.append({"max_dev_mm": round(float(dev[k]), 3), "n_points": int(len(sel) * step),
                          "at_part_mm": [round(float(q[0]), 2), round(float(q[1]), 2)],
                          "nearest_feature": name, "feature_dist_mm": round(dist, 2),
                          "meaning": "modelde fazla malzeme" if dev[k] > 0 else "modelde eksik malzeme"})
        zones.sort(key=lambda z: -abs(z["max_dev_mm"]))

    vis = img.copy()
    extra = (model_mask > 0) & (photo_mask == 0)
    miss = (model_mask == 0) & (photo_mask > 0)
    vis[extra] = (0.4 * vis[extra] + [0, 0, 153]).astype(np.uint8)
    vis[miss] = (0.4 * vis[miss] + [153, 0, 0]).astype(np.uint8)
    cv2.polylines(vis, [np.round(c[::SS]).astype(np.int32).reshape(-1, 1, 2) for c in model_contours],
                  True, (0, 220, 255), 1, cv2.LINE_AA)
    for i, z in enumerate(zones[:8], 1):
        w = o + z["at_part_mm"][0] * ux + z["at_part_mm"][1] * uy
        q = geometry.project([[*w, t]], pose.K, pose.rvec, pose.tvec)[0]
        cv2.circle(vis, tuple(np.round(q).astype(int)), 10, (0, 0, 255) if z["max_dev_mm"] > 0 else (255, 0, 0), 2)
        put_label(vis, f"{i}: {z['max_dev_mm']:+.2f}", q + [12, -12], (0, 0, 180))
    dbg = save_debug(vis, p.path(f"debug/{pid}_karsilastirma.png"), bbox_crop(uv, gray.shape, pad=60))
    a = np.abs(dev)
    return {"photo": pid, "iou": round(iou, 5), "max_abs_dev_mm": round(float(a.max()), 3),
            "p95_abs_dev_mm": round(float(np.percentile(a, 95)), 3),
            "mean_signed_dev_mm": round(float(dev.mean()), 4), "zones": zones[:10], "debug": dbg}


def compare_project(p: Project) -> dict:
    st = p.state().get("last_build")
    if not st:
        return {"ok": False, "error": "önce build_freecad"}
    mesh = trimesh.load(st["stl"], force="mesh")
    feats = _feature_geometry(p)
    res = [compare_photo(p, pid, mesh, feats) for pid, info in p.photos().items() if info.get("processed")]
    done = [r for r in res if "max_abs_dev_mm" in r]
    summary = {"worst_max_dev_mm": max((r["max_abs_dev_mm"] for r in done), default=None),
               "min_iou": min((r["iou"] for r in done), default=None),
               "target_met_0.3mm": bool(done) and all(r["max_abs_dev_mm"] < 0.3 for r in done)}
    p.update_state(last_compare_summary=summary)
    p.log("render_and_compare", summary)
    return {"ok": True, "summary": summary, "photos": res}
