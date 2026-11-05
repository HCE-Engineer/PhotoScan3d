"""Subpixel edge positions along a pixel contour (50 % crossing of a 1D normal profile)."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as nd


@dataclass
class EdgePoints:
    pts: np.ndarray        # (N,2) subpixel image coordinates
    normals: np.ndarray    # (N,2) unit normals, pointing from dark (part) to bright
    contrast: np.ndarray   # (N,) bright - dark level


def refine(gray: np.ndarray, contour: np.ndarray, half_width: float = 4.0, step: float = 0.25,
           min_contrast: float = 25.0) -> EdgePoints:
    """For each contour pixel sample the image along the contour normal and take the
    position where intensity crosses halfway between the local dark and bright levels."""
    g = gray.astype(np.float32)
    c = contour.astype(float)
    n = len(c)
    sm = np.c_[nd.gaussian_filter1d(c[:, 0], 2.0, mode="wrap"), nd.gaussian_filter1d(c[:, 1], 2.0, mode="wrap")]
    tan = np.roll(sm, -1, 0) - np.roll(sm, 1, 0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    nrm = np.c_[tan[:, 1], -tan[:, 0]]
    # orient normals toward the bright side
    probe = lambda p: nd.map_coordinates(g, [p[:, 1], p[:, 0]], order=1, mode="nearest")
    flip = probe(c + 2 * nrm) < probe(c - 2 * nrm)
    nrm[flip] *= -1

    offs = np.arange(-half_width, half_width + 1e-9, step)
    xs = c[:, 0, None] + nrm[:, 0, None] * offs[None]
    ys = c[:, 1, None] + nrm[:, 1, None] * offs[None]
    prof = nd.map_coordinates(g, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(n, -1)
    k = max(2, int(round(1.0 / step)))
    lo = prof[:, :k].mean(1)
    hi = prof[:, -k:].mean(1)
    half = (lo + hi) / 2
    out = np.full(n, np.nan)
    above = prof >= half[:, None]
    for i in range(n):
        idx = np.flatnonzero(above[i, 1:] & ~above[i, :-1])
        if not len(idx):
            continue
        j = idx[np.argmin(np.abs(offs[idx]))]   # crossing nearest the pixel contour
        a, b = prof[i, j], prof[i, j + 1]
        out[i] = offs[j] + step * (half[i] - a) / (b - a) if b != a else offs[j]
    contrast = hi - lo
    ok = np.isfinite(out) & (contrast >= min_contrast)
    pts = c + nrm * np.nan_to_num(out)[:, None]
    return EdgePoints(pts[ok], nrm[ok], contrast[ok])
