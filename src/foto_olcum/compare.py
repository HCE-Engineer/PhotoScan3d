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
        if "circle" in f:
            c = (evaluate(f["circle"]["center"][0], vals), evaluate(f["circle"]["center"][1], vals))
            out.append((f"{f['id']} dış çember", "circle", (c, evaluate(f["circle"]["diameter"], vals) / 2)))
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


# --- an existing STL/CAD against the photos -------------------------------------------

UPS = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}


def _stand(mesh: trimesh.Trimesh, up: str) -> np.ndarray:
    """Vertices rotated so that axis `up` points along +z, resting on z = 0, centred in xy."""
    ax, sgn = UPS[up]
    v = mesh.vertices.copy()
    if ax == 0:
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]) if sgn > 0 else np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    elif ax == 1:
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) if sgn > 0 else np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    else:
        R = np.eye(3) if sgn > 0 else np.diag([1.0, -1.0, -1.0])
    v = v @ R.T
    v[:, 2] -= v[:, 2].min()
    v[:, :2] -= (v[:, :2].min(0) + v[:, :2].max(0)) / 2
    return v


def _place_xy(v, x, y, yaw_deg):
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    out = v.copy()
    out[:, 0] = c * v[:, 0] - s * v[:, 1] + x
    out[:, 1] = s * v[:, 0] + c * v[:, 1] + y
    return out


def _sil(world, faces, pose, crop, shape):
    x0, y0, x1, y1 = crop
    uv = geometry.project(world, pose.K, pose.rvec, pose.tvec)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    tri = np.round((uv[faces] - [x0, y0]) * 4).astype(np.int32)
    for t3 in tri:
        cv2.fillConvexPoly(m, t3, 1, lineType=cv2.LINE_8, shift=2)
    return m


def register_stl(p: Project, pid: str, mesh: trimesh.Trimesh, ups=None) -> dict:
    """Find how the part rests on the sheet: which axis is up, xy position and yaw, by
    maximising silhouette overlap with the photo. The part must rest on a flat face."""
    from scipy.optimize import minimize
    info = p.photos()[pid]
    pose = Pose.from_json(info["pose"])
    mask = cv2.imread(str(p.path(f"masks/{pid}.png")), 0) > 0
    ys, xs = np.nonzero(mask)
    pad = 120
    crop = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(mask.shape[1], xs.max() + pad),
            min(mask.shape[0], ys.max() + pad))
    sub = mask[crop[1]:crop[3], crop[0]:crop[2]]
    if len(mesh.faces) > 6000:
        mesh = mesh.simplify_quadric_decimation(face_count=6000)

    def iou(v, q):
        m = _sil(_place_xy(v, *q), mesh.faces, pose, crop, mask.shape) > 0
        inter = np.count_nonzero(m & sub)
        return inter / max(np.count_nonzero(m | sub), 1)

    best = None
    for up in (ups or UPS):
        v = _stand(mesh, up)
        cz = v[:, 2].max() / 2
        c = geometry.backproject([[xs.mean(), ys.mean()]], pose.K, pose.rvec, pose.tvec, cz)[0]
        for yaw in range(0, 360, 15):
            s = iou(v, (c[0], c[1], yaw))
            if best is None or s > best[0]:
                best = (s, up, (c[0], c[1], yaw))
    s, up, q0 = best
    v = _stand(mesh, up)
    res = minimize(lambda q: -iou(v, q), q0, method="Nelder-Mead",
                   options={"initial_simplex": [q0, q0 + np.r_[1.5, 0, 0], q0 + np.r_[0, 1.5, 0], q0 + np.r_[0, 0, 6]],
                            "xatol": 0.01, "fatol": 1e-5, "maxiter": 400})
    x, y, yaw = res.x
    return {"up": up, "x_mm": float(x), "y_mm": float(y), "yaw_deg": float(yaw % 360), "iou": float(-res.fun),
            "vertices": _place_xy(v, x, y, yaw), "faces": mesh.faces}


def compare_stl(p: Project, stl_path: str, photo_ids=None, up: str | None = None) -> dict:
    """Register an existing STL to each photo and report where the real part deviates."""
    mesh = trimesh.load(stl_path, force="mesh")
    out = []
    for pid, info in p.photos().items():
        if photo_ids and pid not in photo_ids or not info.get("processed"):
            continue
        reg = register_stl(p, pid, mesh, [up] if up else None)
        placed = trimesh.Trimesh(reg.pop("vertices"), reg.pop("faces"), process=False)
        p.update_photo(pid, part_frame={"origin_frame_mm": [0.0, 0.0], "x_dir": [1.0, 0.0], "origin": "altlık"})
        r = compare_photo(p, pid, placed, [])
        r["registration"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in reg.items()}
        out.append(r)
    p.log("compare_stl", {"stl": stl_path, "photos": [r["photo"] for r in out]})
    return {"ok": True, "photos": out}
