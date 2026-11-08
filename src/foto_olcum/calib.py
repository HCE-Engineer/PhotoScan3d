"""Camera intrinsics (printed ChArUco sheet or the pattern on a monitor), fallback from
EXIF, and per-photo pose from the ChArUco sheet.

Images are always read in raw sensor orientation (EXIF rotation ignored) so that portrait
and landscape shots share one calibration.
"""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config
from .board import Sheet, detect

CHARUCO_DICT = cv2.aruco.DICT_5X5_100
CHARUCO_SQUARES = (11, 7)


def imread(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise ValueError(f"görüntü okunamadı: {path}")
    return img


def charuco_board():
    d = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    # Intrinsics do not depend on the board's physical size, so units are arbitrary.
    return cv2.aruco.CharucoBoard(CHARUCO_SQUARES, 1.0, 0.75, d)


def make_pattern(out: Path, screen_w: int = 1920, screen_h: int = 1080) -> Path:
    img = charuco_board().generateImage((screen_w, screen_h), marginSize=max(20, screen_h // 30))
    cv2.imwrite(str(out), img)
    return out


@dataclass
class Camera:
    name: str
    image_size: list          # [w, h]
    K: list
    dist: list
    rms_px: float
    n_images: int
    source: str               # "charuco" | "exif" | "default"
    f_sigma_rel: float        # relative focal-length uncertainty
    created: str = ""

    def save(self):
        p = config.calibration_dir() / f"{self.name}.json"
        p.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, name):
        p = config.calibration_dir() / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"kalibrasyon yok: {name}")
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    def for_size(self, w, h) -> tuple[np.ndarray, np.ndarray]:
        """K, dist for an image of size (w, h); scales K if the photo was resized."""
        cw, ch = self.image_size
        if (w, h) == (cw, ch):
            return np.array(self.K), np.array(self.dist)
        if abs(w / h - cw / ch) > 1e-3:
            raise ValueError(f"fotoğraf boyutu {w}x{h} kalibrasyonla ({cw}x{ch}) aynı oranda değil; "
                             "aynı telefon, aynı zoom ve aynı çözünürlükle çekin")
        s = w / cw
        K = np.array(self.K) * [[s, 1, s], [1, s, s], [1, 1, 1]]
        return K, np.array(self.dist)


def calibrate(paths: list[Path], name: str, sheet: Sheet | None = None) -> dict:
    """Photos of the printed sheet (sheet given) or of the monitor pattern (sheet None)."""
    board = sheet.board() if sheet else charuco_board()
    det = cv2.aruco.CharucoDetector(board)
    objs, imgs, used, skipped, size = [], [], [], [], None
    for p in paths:
        g = cv2.cvtColor(imread(p), cv2.COLOR_BGR2GRAY)
        if size is None:
            size = g.shape[::-1]
        elif g.shape[::-1] != size:
            skipped.append({"file": str(p), "reason": f"boyut farklı {g.shape[::-1]} != {size}"})
            continue
        cc, ci, _, _ = det.detectBoard(g)
        if ci is None or len(ci) < 8:
            skipped.append({"file": str(p), "reason": f"yetersiz köşe ({0 if ci is None else len(ci)})"})
            continue
        o, i = board.matchImagePoints(cc, ci)
        objs.append(o.astype(np.float32))
        imgs.append(i.astype(np.float32))
        used.append(str(p))
    if len(objs) < 6:
        raise ValueError(f"en az 6 kullanılabilir fotoğraf gerekli, {len(objs)} var. Atlananlar: {skipped}")
    # k3 left out: a phone calibration rarely covers the image corners, and a free k3 then
    # invents large corner distortion without improving the measured area (real S24 FE
    # photos: same hold-out accuracy, 100 px of made-up corner warp)
    def run(o_, i_):
        rms_, K_, d_, rv_, tv_ = cv2.calibrateCamera(o_, i_, size, None, None, flags=cv2.CALIB_FIX_K3)
        per_ = []
        for o, i, r, t in zip(o_, i_, rv_, tv_):
            proj, _ = cv2.projectPoints(o, r, t, K_, d_)
            per_.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))))
        return rms_, K_, d_, per_

    # Phone autofocus/OIS shift the intrinsics a little per shot; drop the shots that fit
    # clearly worse than the rest (on real S24 FE photos this cut the worst hold-out
    # distance error from 0.53 to 0.44 mm).
    dropped = []
    rms, K, dist, per = run(objs, imgs)
    while len(objs) > 10:
        w = int(np.argmax(per))
        if per[w] <= max(0.8, 1.5 * float(np.median(per))):
            break
        dropped.append({"file": used[w], "rms_px": round(per[w], 3)})
        for lst in (objs, imgs, used):
            lst.pop(w)
        rms, K, dist, per = run(objs, imgs)
    cam = Camera(name, list(size), K.tolist(), dist.ravel().tolist(), float(rms), len(objs),
                 "charuco", 0.003, time.strftime("%Y-%m-%d %H:%M"))
    path = cam.save()
    warn = []
    if rms > 0.5:
        warn.append(f"reprojeksiyon hatası yüksek ({rms:.2f} px > 0.5): bulanık ya da odak/zoom "
                    "değişen fotoğrafları çıkarıp tekrar deneyin; desenin fotoğraf kenar ve köşelerine de "
                    "geldiği kareler ekleyin")
    worst = sorted(zip(per, used), reverse=True)[:3]
    return {"rms_px": float(rms), "n_used": len(objs), "skipped": skipped, "dropped_outliers": dropped,
            "file": str(path),
            "image_size": list(size), "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "worst_images": [{"file": u, "rms_px": e} for e, u in worst], "warnings": warn}


def camera_from_exif(path, w, h) -> Camera:
    """Fallback intrinsics: focal length from EXIF 35 mm equivalent, no distortion."""
    f35 = None
    try:
        exif = Image.open(path).getexif()
        f35 = exif.get_ifd(0x8769).get(41989)
    except Exception:
        pass
    diag = float(np.hypot(w, h))
    if f35:
        f, src, sig = float(f35) / 43.27 * diag, "exif", 0.03
    else:
        f, src, sig = 0.6 * diag, "default", 0.3        # ~26 mm equivalent main camera
    K = [[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]
    return Camera(f"_tahmini_{src}", [w, h], K, [0, 0, 0, 0, 0], float("nan"), 0, src, sig)


@dataclass
class Pose:
    K: np.ndarray            # intrinsics of the undistorted image
    rvec: np.ndarray
    tvec: np.ndarray
    rms_px: float
    n_corners: int
    corner_ids: list
    scale_sigma_rel: float   # relative scale uncertainty from pose (bootstrap)
    shift_sigma_mm: float    # absolute position uncertainty (bootstrap, window RMS)

    @property
    def R(self):
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def camera_center(self):
        return (-self.R.T @ self.tvec.reshape(3)).ravel()

    def to_json(self):
        c = self.camera_center
        return {"rms_px": round(self.rms_px, 3), "n_corners": self.n_corners,
                "corner_ids": self.corner_ids, "camera_center_mm": [round(v, 1) for v in c],
                "scale_sigma_rel": self.scale_sigma_rel, "shift_sigma_mm": round(self.shift_sigma_mm, 4),
                "rvec": self.rvec.ravel().tolist(), "tvec": self.tvec.ravel().tolist(),
                "K": self.K.tolist()}

    @classmethod
    def from_json(cls, d):
        return cls(np.array(d["K"]), np.array(d["rvec"]).reshape(3, 1), np.array(d["tvec"]).reshape(3, 1),
                   d["rms_px"], d["n_corners"], d["corner_ids"], d["scale_sigma_rel"], d["shift_sigma_mm"])


def _solve(obj, img, K):
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP başarısız")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    return rvec, tvec


MIN_CORNERS = 12


def estimate_pose(gray_undist, K, sheet: Sheet, rng_seed=0) -> Pose:
    img, ids = detect(gray_undist, sheet)
    if len(ids) < MIN_CORNERS:
        raise ValueError(f"altlıkta yetersiz ChArUco köşesi: {len(ids)} bulundu (en az {MIN_CORNERS}, "
                         "tercihen 40+). Kağıdın tamamı kadrajda, düz ve aydınlık olmalı.")
    obj = sheet.corners_mm(ids)
    rvec, tvec = _solve(obj, img, K)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))

    # Bootstrap over corners: how much do window points and distances move?
    from .geometry import backproject
    wx0, wy0, wx1, wy1 = sheet.window
    gx, gy = np.meshgrid(np.linspace(wx0, wx1, 5), np.linspace(wy0, wy1, 5))
    grid = np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)]
    gimg, _ = cv2.projectPoints(grid, rvec, tvec, K, None)
    gimg = gimg.reshape(-1, 2)
    rng = np.random.default_rng(rng_seed)
    shifts, scales = [], []
    diag = np.hypot(wx1 - wx0, wy1 - wy0)
    for _ in range(30):
        pick = rng.choice(len(ids), len(ids), replace=True)
        if len(set(pick)) < 6:
            continue
        o, im = obj[pick], img[pick]
        try:
            rv, tv = _solve(o, im, K)
        except (ValueError, cv2.error):
            continue
        p = backproject(gimg, K, rv, tv, 0.0)
        shifts.append(np.sqrt(np.mean(np.sum((p - grid[:, :2]) ** 2, 1))))
        scales.append(np.linalg.norm(p[0] - p[-1]) / diag - 1)
    shift = float(np.mean(shifts)) if shifts else float("nan")
    scale = float(np.std(scales)) if len(scales) > 5 else 0.01
    return Pose(K, rvec, tvec, rms, len(ids), ids.tolist(), scale, shift)
