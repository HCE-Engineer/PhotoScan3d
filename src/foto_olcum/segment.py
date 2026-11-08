"""Backlit silhouette: the part is dark on the backlit paper inside the blank middle
of the ChArUco sheet."""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Silhouette:
    mask: np.ndarray                 # uint8 0/255, part = 255
    roi: np.ndarray                  # uint8 0/255, usable window area
    outer: np.ndarray                # (N,2) int pixel contour of the part
    holes: list                      # list of (N,2) int contours (bright regions inside part)
    threshold: float
    levels: tuple                    # (dark, bright) medians
    warnings: list = field(default_factory=list)


def roi_mask(shape, window_px: np.ndarray) -> np.ndarray:
    roi = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(roi, [np.round(window_px).astype(np.int32)], 255)
    return roi


def flatten(gray: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Divide out the slowly varying backlight brightness (a laptop LCD seen at an angle
    darkens strongly across the screen), so one global threshold separates part and
    background everywhere. The background is estimated by a large grey closing, which
    fills in the dark part with the surrounding bright screen."""
    s = 8
    small = cv2.resize(gray, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(roi[::s, ::s])
    span = min(np.ptp(xs), np.ptp(ys)) if len(xs) else min(small.shape)
    k = max(15, int(span * 0.45)) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), k / 6)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(gray.astype(np.float32) / np.maximum(bg, 1.0) * 200.0, 0, 255).astype(np.uint8)


def backlit(gray: np.ndarray, roi: np.ndarray, min_hole_px: int = 30) -> Silhouette:
    warnings = []
    flat = flatten(gray, roi)
    vals = flat[roi > 0]
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((flat < thr) & (roi > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        raise ValueError("pencerede koyu bir parça bulunamadı (arka ışık açık mı, parça pencerede mi?)")
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    part = order[0]
    roi_area = int((roi > 0).sum())
    others = [int(stats[i, cv2.CC_STAT_AREA]) for i in order[1:] if stats[i, cv2.CC_STAT_AREA] > 0.002 * roi_area]
    if others:
        warnings.append(f"pencerede başka koyu nesneler var ({len(others)} adet, en büyüğü {max(others)} px); "
                        "yok sayıldı. Toz/gölge ise temizleyin.")
    mask = (lab == part).astype(np.uint8) * 255
    edge = cv2.morphologyEx(roi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if (mask & edge).any():
        warnings.append("parça altlığın boş alanının kenarına (desene) değiyor veya çok yakın: kenar ölçüleri güvenilmez")
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    outer_i = max((i for i in range(len(contours)) if hier[i][3] < 0), key=lambda i: cv2.contourArea(contours[i]))
    holes = [contours[i].reshape(-1, 2) for i in range(len(contours))
             if hier[i][3] == outer_i and cv2.contourArea(contours[i]) >= min_hole_px]
    dark_med = float(np.median(gray[mask > 0]))
    bright_med = float(np.median(gray[(roi > 0) & (mask == 0)]))
    if bright_med - dark_med < 60:
        warnings.append(f"kontrast düşük (koyu {dark_med:.0f}, açık {bright_med:.0f}); "
                        "ekran parlaklığını artırın, ortam ışığını azaltın")
    if bright_med > 250:
        warnings.append("arka plan doymuş (255'e yakın): pozlamayı biraz düşürün, kenarlar şişebilir")
    return Silhouette(mask, roi, contours[outer_i].reshape(-1, 2), holes, float(thr),
                      (dark_med, bright_med), warnings)
