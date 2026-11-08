"""Shared helpers: per-photo loading/undistortion and debug image output."""
from pathlib import Path

import cv2
import numpy as np

from .calib import Camera, camera_from_exif, imread

PREVIEW_MAX = 1024


def camera_for(photo_path, camera_name: str | None, w: int, h: int) -> tuple[np.ndarray, np.ndarray, Camera]:
    if camera_name:
        cam = Camera.load(camera_name)
    else:
        cam = camera_from_exif(photo_path, w, h)
    K, dist = cam.for_size(w, h)
    return K, dist, cam


def undistorted(photo_path, camera_name):
    """(undistorted image, K, camera, raw image, dist)."""
    raw = imread(photo_path)
    h, w = raw.shape[:2]
    K, dist, cam = camera_for(photo_path, camera_name, w, h)
    img = cv2.undistort(raw, K, dist) if np.any(np.abs(dist) > 0) else raw
    return img, K, cam, raw, dist


def save_debug(img: np.ndarray, path: Path, crop=None) -> dict:
    """Save full-resolution debug image (optionally cropped) and a <=1024 px preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if crop is not None:
        x0, y0, x1, y1 = crop
        img = img[max(0, y0):y1, max(0, x0):x1]
    cv2.imwrite(str(path), img)
    s = PREVIEW_MAX / max(img.shape[:2])
    prev = path.with_name(path.stem + "_k.jpg")
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    cv2.imwrite(str(prev), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"full": str(path), "preview": str(prev)}


def bbox_crop(pts: np.ndarray, shape, pad: int = 60):
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    return max(0, x0), max(0, y0), min(shape[1], x1), min(shape[0], y1)


def put_label(img, text, xy, color, scale=0.6):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
