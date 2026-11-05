"""Image (undistorted pixels) <-> frame plane z = h, and the silhouette footprint rule."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

TANGENTIAL = 0.25    # |(p_t - p_0)·n| / |p_t - p_0| below which a convex wall is blended
CONVEX_KAPPA = 0.05  # 1/mm: curvature (radius 20 mm) at which a wall counts as fully curved


def backproject(uv, K, rvec, tvec, z: float) -> np.ndarray:
    """Intersect camera rays through pixels `uv` (N,2) with the frame plane z = const."""
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    t = np.asarray(tvec, float).reshape(3)
    C = -R.T @ t
    uv = np.asarray(uv, float).reshape(-1, 2)
    rays = (R.T @ np.linalg.solve(K, np.c_[uv, np.ones(len(uv))].T)).T
    s = (z - C[2]) / rays[:, 2]
    return C[:2] + rays[:, :2] * s[:, None]


def project(xyz, K, rvec, tvec) -> np.ndarray:
    p, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), np.asarray(rvec, float),
                             np.asarray(tvec, float), K, None)
    return p.reshape(-1, 2)


def footprint(uv, normals_px, K, rvec, tvec, thickness: float, closed: bool = True):
    """Footprint of a vertical-walled 2.5D part's silhouette points.

    A silhouette ray grazes either the top edge (z = t) or the bottom edge (z = 0) of the
    wall. The real wall position is whichever of the two plane intersections lies further
    *inside* the material, i.e. smaller along the outward normal (material -> background).
    Returns (points_mm, outward_normals_mm, ambiguity_mm = |p_t - p_0|, used_top mask).
    """
    p0 = backproject(uv, K, rvec, tvec, 0.0)
    pt = backproject(uv, K, rvec, tvec, thickness)
    q = backproject(np.asarray(uv) + np.asarray(normals_px), K, rvec, tvec, thickness)
    n = q - pt
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    along_n = np.einsum("ij,ij->i", pt - p0, n)
    use_top = along_n < 0
    amb = np.linalg.norm(pt - p0, axis=1)
    # The chosen edge point is a real 3D point on the wall's top or bottom edge, so its
    # footprint is exact. One exception: where a *convex* curved wall (outer fillet) is seen
    # tangentially, the silhouette is a vertical generator of the wall whose footprint lies
    # between the edges; there we ramp through the midpoint. Concave walls (holes, inner
    # fillets) instead show a sharp crossing of the top and bottom arcs, which the plain
    # choice gets right. s and the convexity weight are smoothed along the closed contour
    # so normal noise cannot make points jitter tangentially and break contour order.
    pts = np.where(use_top[:, None], pt, p0)
    if not closed or len(uv) < 20:
        return pts, n, amb, use_top
    s = gaussian_filter1d(along_n / np.maximum(amb, 1e-9), 5.0, mode="wrap")
    sm = np.c_[gaussian_filter1d(pt[:, 0], 5.0, mode="wrap"), gaussian_filter1d(pt[:, 1], 5.0, mode="wrap")]
    seg = np.linalg.norm(np.roll(sm, -1, 0) - sm, axis=1).mean()
    k = max(3, int(round(0.5 / max(seg, 1e-6))))             # ~0.5 mm baseline
    curv = (np.roll(sm, -k, 0) + np.roll(sm, k, 0) - 2 * sm) / (k * seg) ** 2
    convex = gaussian_filter1d(np.clip(-np.einsum("ij,ij->i", curv, n) / CONVEX_KAPPA, 0, 1), 5.0, mode="wrap")
    w_ramp = np.clip(0.5 - s / (2 * TANGENTIAL), 0.0, 1.0)
    w_top = convex * w_ramp + (1 - convex) * use_top
    pts = w_top[:, None] * pt + (1 - w_top[:, None]) * p0
    return pts, n, amb, use_top


def mm_per_px(K, rvec, tvec, uv, z: float) -> np.ndarray:
    a = backproject(uv, K, rvec, tvec, z)
    b = backproject(np.asarray(uv, float) + [1.0, 0.0], K, rvec, tvec, z)
    return np.linalg.norm(b - a, axis=1)
