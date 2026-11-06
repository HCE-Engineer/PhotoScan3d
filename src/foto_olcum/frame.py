"""3D-printed reference frame: ArUco markers around an open window.

Frame coordinates (mm): origin at the outer lower-left corner, x right, y up, z up
(away from the backlight screen). The frame lies on the screen at z=0.

Printed as a thin white base with black marker/body regions on top. With a backlight the
thin (white) regions glow and the thick (black) ones stay dark, so it also works in a
single light-coloured filament; a filament change at `base_mm` gives the best contrast.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh

from . import config

DICT_NAME = "DICT_4X4_50"
RES = 0.5  # raster resolution, mm
MIN_BLACK_MM = 1.5  # narrowest printable black feature (0.6 mm nozzle: ~2 extrusion widths)


@dataclass
class Frame:
    name: str
    outer_w: float
    outer_h: float
    marker_mm: float
    quiet_mm: float
    rim_mm: float
    base_mm: float
    black_mm: float
    markers: dict = field(default_factory=dict)       # id -> [x0, y0] lower-left, mm
    measured_window: list | None = None               # [w, h] by caliper
    dictionary: str = DICT_NAME

    @property
    def band(self):
        return self.rim_mm * 2 + self.quiet_mm * 2 + self.marker_mm

    @property
    def window(self):
        """[x0, y0, x1, y1] of the open window (nominal)."""
        b = self.band
        return [b, b, self.outer_w - b, self.outer_h - b]

    @property
    def z_marker(self):
        """Height used for marker corners: middle of the black layer."""
        return self.base_mm + self.black_mm / 2

    @property
    def scale(self):
        if not self.measured_window:
            return 1.0, 1.0
        x0, y0, x1, y1 = self.window
        return self.measured_window[0] / (x1 - x0), self.measured_window[1] / (y1 - y0)

    @property
    def scale_sigma_rel(self):
        """Relative scale uncertainty: printer XY error if unmeasured, else caliper."""
        if not self.measured_window:
            return 0.004
        return 0.03 / min(self.measured_window)

    def _apply_scale(self, x, y):
        x0, y0, x1, y1 = self.window
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx, sy = self.scale
        return cx + (x - cx) * sx, cy + (y - cy) * sy

    def marker_corners(self, mid: int) -> np.ndarray:
        """4x3 corners in ArUco order (TL, TR, BR, BL of the marker image), frame mm."""
        x0, y0 = self.markers[str(mid)]
        s = self.marker_mm
        pts = [(x0, y0 + s), (x0 + s, y0 + s), (x0 + s, y0), (x0, y0)]
        return np.array([(*self._apply_scale(x, y), self.z_marker) for x, y in pts], float)

    def window_poly(self, shrink: float = 0.0) -> np.ndarray:
        x0, y0, x1, y1 = self.window
        pts = [(x0 + shrink, y0 + shrink), (x1 - shrink, y0 + shrink),
               (x1 - shrink, y1 - shrink), (x0 + shrink, y1 - shrink)]
        return np.array([self._apply_scale(x, y) for x, y in pts], float)

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, name_or_path) -> "Frame":
        p = Path(name_or_path)
        if not p.suffix:
            p = config.frames_dir() / name_or_path / "frame.json"
        if not p.exists():
            raise FileNotFoundError(f"çerçeve bulunamadı: {name_or_path}")
        return cls(**json.loads(p.read_text(encoding="utf-8")))


def aruco_dict():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))


def layout(name, outer_w=210.0, outer_h=210.0, marker_mm=24.0, quiet_mm=4.0, rim_mm=2.0,
           base_mm=0.6, black_mm=0.6) -> Frame:
    f = Frame(name, outer_w, outer_h, marker_mm, quiet_mm, rim_mm, base_mm, black_mm)
    slot = marker_mm + quiet_mm
    b = f.band
    if outer_w - 2 * b < 20 or outer_h - 2 * b < 20:
        raise ValueError("çerçeve çok küçük: pencere en az 20 mm olmalı")

    def spots(length):
        n = int((length - 2 * rim_mm - quiet_mm) // slot)
        free = length - 2 * rim_mm - n * marker_mm
        gap = free / (n + 1)
        return [round((rim_mm + gap * (i + 1) + marker_mm * i) / RES) * RES for i in range(n)]

    off = rim_mm + quiet_mm
    pos = []
    for x in spots(outer_w):                       # bottom and top bands, full width
        pos += [(x, off), (x, outer_h - off - marker_mm)]
    inner = outer_h - 2 * b                        # left and right bands, between them
    n = int((inner - quiet_mm) // slot)
    if n > 0:
        gap = (inner - n * marker_mm) / (n + 1)
        for i in range(n):
            y = round((b + gap * (i + 1) + marker_mm * i) / RES) * RES
            pos += [(off, y), (outer_w - off - marker_mm, y)]
    if len(pos) > 50:
        raise ValueError("çok fazla marker (en fazla 50)")
    f.markers = {str(i): [float(x), float(y)] for i, (x, y) in enumerate(sorted(pos, key=lambda p: (p[1], p[0])))}
    return f


def black_raster(f: Frame) -> np.ndarray:
    """Boolean raster (rows = y from bottom) of the black layer, RES mm per cell."""
    nx, ny = int(round(f.outer_w / RES)), int(round(f.outer_h / RES))
    r = np.ones((ny, nx), bool)
    x0, y0, x1, y1 = (int(round(v / RES)) for v in f.window)
    r[y0:y1, x0:x1] = False
    q, s = f.quiet_mm, f.marker_mm
    d = aruco_dict()
    for mid, (mx, my) in f.markers.items():
        a, b = int(round((mx - q) / RES)), int(round((my - q) / RES))
        w = int(round((s + 2 * q) / RES))
        r[b:b + w, a:a + w] = False
        img = cv2.aruco.generateImageMarker(d, int(mid), 6)       # 6x6 cells incl. border
        cells = int(round(s / 6 / RES))
        if abs(cells * 6 * RES - s) > 1e-6:
            raise ValueError("marker_mm, 6 * 0.5 mm'nin katı olmalı (ör. 24, 27, 30)")
        big = np.kron(img == 0, np.ones((cells, cells), bool))[::-1]  # flip: image top = +y
        mx0, my0 = int(round(mx / RES)), int(round(my / RES))
        r[my0:my0 + big.shape[0], mx0:mx0 + big.shape[1]] = big
    # drop black slivers narrower than MIN_BLACK_MM (e.g. ribs between neighbouring quiet
    # zones): a 0.4-0.6 mm nozzle cannot print them. Marker cells are far wider and survive.
    k = int(np.ceil(MIN_BLACK_MM / RES))
    r = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(bool)
    return r


def _rects(mask: np.ndarray):
    """Greedy cover of a boolean raster with axis-aligned rectangles (row runs merged down)."""
    m = mask.copy()
    out = []
    ny, nx = m.shape
    for y in range(ny):
        x = 0
        while x < nx:
            if not m[y, x]:
                x += 1
                continue
            x1 = x
            while x1 < nx and m[y, x1]:
                x1 += 1
            y1 = y + 1
            while y1 < ny and m[y1, x:x1].all():
                y1 += 1
            m[y:y1, x:x1] = False
            out.append((x, y, x1, y1))
            x = x1
    return out


def _boxes(rects, z0, z1):
    meshes = []
    for x0, y0, x1, y1 in rects:
        ext = [(x1 - x0) * RES, (y1 - y0) * RES, z1 - z0]
        c = [(x0 + x1) * RES / 2, (y0 + y1) * RES / 2, (z0 + z1) / 2]
        meshes.append(trimesh.creation.box(extents=ext, transform=trimesh.transformations.translation_matrix(c)))
    return trimesh.util.concatenate(meshes)


def build(f: Frame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    nx, ny = int(round(f.outer_w / RES)), int(round(f.outer_h / RES))
    base = np.ones((ny, nx), bool)
    x0, y0, x1, y1 = (int(round(v / RES)) for v in f.window)
    base[y0:y1, x0:x1] = False
    black = black_raster(f)
    base_mesh = _boxes(_rects(base), 0.0, f.base_mm)
    black_mesh = _boxes(_rects(black), f.base_mm, f.base_mm + f.black_mm)
    files = {
        "stl_all": out_dir / f"{f.name}.stl",
        "stl_base_white": out_dir / f"{f.name}_beyaz_taban.stl",
        "stl_black": out_dir / f"{f.name}_siyah.stl",
        "preview_png": out_dir / f"{f.name}_onizleme.png",
        "config": out_dir / "frame.json",
    }
    trimesh.util.concatenate([base_mesh, black_mesh]).export(files["stl_all"])
    base_mesh.export(files["stl_base_white"])
    black_mesh.export(files["stl_black"])
    prev = np.full((ny, nx), 255, np.uint8)
    prev[black] = 0
    prev[y0:y1, x0:x1] = 200
    cv2.imwrite(str(files["preview_png"]), cv2.resize(prev[::-1], (nx * 2, ny * 2), interpolation=cv2.INTER_NEAREST))
    f.save(files["config"])
    return {k: str(v) for k, v in files.items()}
