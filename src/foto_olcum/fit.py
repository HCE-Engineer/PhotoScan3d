"""Line / circle fits with statistical uncertainty, and contour segmentation (mm).

Edge samples are about one pixel apart and correlated over a few pixels by blur and
contour smoothing, so statistical errors use an effective sample count N / CORR.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as nd
from scipy.optimize import minimize_scalar

CORR = 3.0


@dataclass
class Line:
    c: np.ndarray       # centroid
    u: np.ndarray       # unit direction
    n: np.ndarray       # unit normal
    rms: float
    sigma_offset: float # std of the line position along n at the centroid
    sigma_angle: float  # rad
    length: float
    idx: tuple          # (start, end) sample indices in the resampled contour

    def point_sigma(self, p) -> float:
        """Std of the line's normal position at point p (offset + rotation about centroid)."""
        s = float(np.dot(np.asarray(p) - self.c, self.u))
        return float(np.hypot(self.sigma_offset, s * self.sigma_angle))


@dataclass
class Circle:
    c: np.ndarray
    r: float
    rms: float
    sigma_c: float
    sigma_r: float
    n: int


def fit_line(p: np.ndarray, idx=(0, 0)) -> Line:
    c = p.mean(0)
    q = p - c
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    u = vt[0]
    nrm = np.array([-u[1], u[0]])
    d = q @ nrm
    t = q @ u
    rms = float(np.sqrt(np.mean(d ** 2)))
    neff = max(len(p) / CORR, 2.0)
    sig = rms * np.sqrt(CORR) if rms > 0 else 1e-6
    return Line(c, u, nrm, rms, float(sig / np.sqrt(len(p))), float(sig / np.sqrt(np.sum(t ** 2))),
                float(t.max() - t.min()), idx)


def fit_circle(p: np.ndarray) -> Circle:
    """Taubin algebraic fit followed by Gauss-Newton refinement of the geometric error."""
    m = p.mean(0)
    x, y = (p - m).T
    z = x * x + y * y
    zm = z.mean()
    z0 = (z - zm) / (2 * np.sqrt(zm))
    _, _, vt = np.linalg.svd(np.c_[z0, x, y], full_matrices=False)
    a = vt[2]
    a0 = a[0] / (2 * np.sqrt(zm))
    a = np.array([a0, a[1], a[2], -zm * a0])
    cx, cy = -a[1] / (2 * a[0]), -a[2] / (2 * a[0])
    r = np.sqrt(a[1] ** 2 + a[2] ** 2 - 4 * a[0] * a[3]) / (2 * abs(a[0]))
    th = np.array([cx, cy, r])
    for _ in range(10):
        dx, dy = x - th[0], y - th[1]
        d = np.hypot(dx, dy)
        res = d - th[2]
        J = np.c_[-dx / d, -dy / d, -np.ones_like(d)]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        th += step
        if np.abs(step).max() < 1e-10:
            break
    dx, dy = x - th[0], y - th[1]
    d = np.hypot(dx, dy)
    res = d - th[2]
    J = np.c_[-dx / d, -dy / d, -np.ones_like(d)]
    rms = float(np.sqrt(np.mean(res ** 2)))
    cov = (max(rms, 1e-6) ** 2) * CORR * np.linalg.pinv(J.T @ J)
    return Circle(th[:2] + m, float(th[2]), rms, float(np.sqrt((cov[0, 0] + cov[1, 1]) / 2)),
                  float(np.sqrt(cov[2, 2])), len(p))


def resample_closed(p: np.ndarray, ds: float) -> np.ndarray:
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, s[-1], ds)
    return np.c_[np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])]


def intersect(a: Line, b: Line):
    A = np.array([a.u, -b.u]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s, _ = np.linalg.solve(A, b.c - a.c)
    return a.c + s * a.u


@dataclass
class Corner:
    p: np.ndarray          # virtual sharp corner (intersection of the two lines)
    sigma_p: float
    radius: float          # 0 = sharp
    sigma_r: float
    turn_deg: float        # signed turn angle (+ = left)
    rms: float             # residual of the fillet model on the gap points
    explained: bool        # False -> gap is not a simple sharp/tangent-fillet corner


def _fillet_fit(gap: np.ndarray, a: Line, b: Line, v: np.ndarray, turn: float):
    """1D least squares for a fillet radius tangent to both lines at corner v.

    The model curve is line a up to its tangent point, the arc, then line b, so r = 0 is
    the sharp corner and points on the straight parts are not forced onto the circle."""
    bis = (-a.u + b.u)
    bis /= np.linalg.norm(bis)
    half_int = (np.pi - abs(turn)) / 2                     # half interior angle
    sa = gap - v

    def resid(r):
        dt = r / np.tan(half_int)                          # vertex -> tangent point
        c = v + bis * r / np.sin(half_int)
        on_a = sa @ a.u < -dt
        on_b = sa @ b.u > dt
        d = np.abs(np.linalg.norm(gap - c, axis=1) - r)
        d = np.where(on_a, np.abs(sa @ a.n), d)
        d = np.where(on_b, np.abs(sa @ b.n), d)
        return d

    def cost(r):
        return float(np.mean(resid(r) ** 2))

    la, lb = a.length, b.length
    rmax = max(1e-3, 0.95 * min(la, lb) * np.tan(half_int)) if la and lb else 50.0
    res = minimize_scalar(cost, bounds=(0.0, rmax), method="bounded", options={"xatol": 1e-4})
    r = float(res.x)
    h = 1e-3
    curv = (cost(r + h) - 2 * cost(r) + cost(max(r - h, 0))) / h ** 2
    rms = float(np.sqrt(res.fun))
    sigma_r = float(np.sqrt(2 * max(rms, 1e-6) ** 2 * CORR / max(curv * len(gap), 1e-12)))
    return r, sigma_r, rms


def segment(points: np.ndarray, ds: float, noise_mm: float, min_line_mm: float = 2.0,
            kappa_straight: float = 0.03, sharp_r: float = 0.3):
    """Split a closed contour (mm, ordered) into straight lines and corners.

    Returns (lines, corners, samples). corners[i] joins lines[i] and lines[i+1]."""
    p = resample_closed(points, ds)
    m = len(p)
    k = max(2, int(round(0.8 / ds)))
    d = np.roll(p, -k, 0) - np.roll(p, k, 0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    # curvature from a wrap-safe smoothed derivative of the tangent angle
    dth = np.angle(np.exp(1j * (np.roll(th, -k) - np.roll(th, k)))) / (2 * k * ds)
    kap = nd.uniform_filter1d(dth, max(1, k), mode="wrap")
    # most of a prismatic outline is straight, so the MAD of curvature is its noise level
    kap_noise = 1.4826 * np.median(np.abs(kap - np.median(kap)))
    straight = np.abs(kap) < max(kappa_straight, 5 * kap_noise)
    if straight.all() or not straight.any():
        return [], [], p
    start = int(np.flatnonzero(~straight)[0])          # roll so index 0 is non-straight
    p = np.roll(p, -start, 0)
    straight = np.roll(straight, -start)
    runs, i = [], 0
    while i < m:
        if straight[i]:
            j = i
            while j < m and straight[j]:
                j += 1
            if (j - i) * ds >= min_line_mm:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        return [], [], p
    trim = max(1, int(round(0.3 / ds)))
    lines = [fit_line(p[a + trim:b - trim] if b - a > 2 * trim + 3 else p[a:b], (a, b)) for a, b in runs]
    # orient lines along the contour direction
    for ln, (a, b) in zip(lines, runs):
        if np.dot(p[b - 1] - p[a], ln.u) < 0:
            ln.u, ln.n = -ln.u, -ln.n
    corners = []
    for i, a in enumerate(lines):
        b = lines[(i + 1) % len(lines)]
        ia, ib = runs[i][1], runs[(i + 1) % len(runs)][0]
        gap = p[ia:ib] if ib > ia else np.vstack([p[ia:], p[:ib]])
        turn = float(np.arctan2(a.u[0] * b.u[1] - a.u[1] * b.u[0], np.dot(a.u, b.u)))
        v = intersect(a, b)
        if v is None or abs(turn) < np.radians(3):
            corners.append(Corner(p[ia], np.nan, 0.0, np.nan, np.degrees(turn), np.nan, False))
            continue
        sv = float(np.hypot(a.point_sigma(v), b.point_sigma(v)) / max(np.sin(abs(turn)), 0.2))
        ctx = np.vstack([p[max(ia - trim, 0):ia], gap, p[ib:ib + trim]]) if len(gap) else p[ia - 1:ib + 1]
        if len(ctx) < 4:
            corners.append(Corner(v, sv, 0.0, 0.0, np.degrees(turn), 0.0, True))
            continue
        r, sr, rms = _fillet_fit(ctx, a, b, v, turn)
        explained = rms < max(4 * noise_mm, 0.05)
        # a fillet only shows as its sagitta at the corner; at shallow turns even a large
        # radius hides below the edge noise
        sag_per_r = 1 / np.cos(abs(turn) / 2) - 1
        r_visible = max(sharp_r, 3 * noise_mm / max(sag_per_r, 1e-9))
        if r < r_visible or r < 2 * sr:     # not distinguishable from a sharp corner
            r, sr = 0.0, max(r_visible / 2, sr)
        corners.append(Corner(v, sv, r, sr, np.degrees(turn), rms, explained))
    return lines, corners, p
