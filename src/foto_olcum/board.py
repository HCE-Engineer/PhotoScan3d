"""ChArUco base with a blank middle area for the part: printed on A4 paper, or shown
full-screen on a monitor laid flat (the screen is then both backlight and pattern).

Sheet coordinates (mm): origin at the board's lower-left corner, x right, y up, z up
(toward the camera). The part sits on z = 0. On paper the pattern is at z = 0; on a screen
the pixels lie `glass_mm` below the glass the part sits on, so the pattern is at -glass_mm.
OpenCV's ChArUco object points run x right, y *down* from the top-left, so
y_sheet = height - y_board.

On a white backlight the paper glows, printed squares stay dark, and the part in the blank
middle shows as a silhouette. ChArUco chessboard corners are located far more precisely
than single ArUco marker corners. The same sheet also serves for camera calibration.

Printers scale by ~0.5-1 % and a screen's EDID size is rounded to whole mm; the user
measures A-B (x) and A-C (y) with a caliper and the object points are scaled accordingly.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

DICT_NAME = "DICT_4X4_250"
DPI = 600                        # 12.7 mm = 300 px exactly
PAGE_MM = (210.0, 297.0)
REF_SQUARES = 10                 # caliper check span: corner A -> B (x), A -> C (y)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass
class Sheet:
    name: str
    squares_x: int = 15
    squares_y: int = 22
    square_mm: float = 12.7
    marker_mm: float = 9.525
    border: int = 2               # squares of pattern around the blank middle (paper)
    measured_ref: list | None = None   # [A-B, A-C] in mm, by caliper
    dictionary: str = DICT_NAME
    kind: str = "paper"           # "paper" | "screen"
    blank: list | None = None     # [x0, y0, x1, y1] blank area in squares (y up); default: border
    ref_origin: list | None = None     # A in squares; default [1, 1]
    ref_squares: int = REF_SQUARES     # A-B and A-C span in squares
    glass_mm: float = 0.0         # pattern plane below the part's resting surface
    glass_sigma_mm: float = 0.0
    square_px: int = 0            # screen: square size in screen pixels
    screen: dict | None = None    # screen: connector, resolution, EDID size

    @property
    def width(self):
        return self.squares_x * self.square_mm

    @property
    def height(self):
        return self.squares_y * self.square_mm

    @property
    def blank_sq(self):
        if self.blank:
            return self.blank
        b = self.border
        return [b, b, self.squares_x - b, self.squares_y - b]

    @property
    def window(self):
        """[x0, y0, x1, y1] of the blank middle area (nominal sheet mm)."""
        return [v * self.square_mm for v in self.blank_sq]

    @property
    def ref_nominal(self):
        return self.ref_squares * self.square_mm

    @property
    def ref_points(self):
        """Caliper points A, B (= A + n squares in x), C (= A + n squares in y)."""
        a = np.array(self.ref_origin or [1.0, 1.0], float) * self.square_mm
        d = self.ref_nominal
        return {"A": a, "B": a + [d, 0], "C": a + [0, d]}

    @property
    def scale(self):
        if not self.measured_ref:
            return 1.0, 1.0
        return self.measured_ref[0] / self.ref_nominal, self.measured_ref[1] / self.ref_nominal

    @property
    def scale_sigma_rel(self):
        """Printer/EDID scale error if unmeasured, else the caliper reading."""
        if not self.measured_ref:
            return 0.01 if self.kind == "paper" else 0.003
        return 0.05 / self.ref_nominal

    def board(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_mm, self.marker_mm, d)

    def _apply_scale(self, xy):
        sx, sy = self.scale
        c = np.array([self.width / 2, self.height / 2])
        return c + (np.asarray(xy, float) - c) * [sx, sy]

    def corners_mm(self, ids) -> np.ndarray:
        """Sheet coordinates (N,3) of ChArUco corner ids, printer scale applied."""
        obj = self.board().getChessboardCorners()[np.asarray(ids).ravel()]
        xy = np.c_[obj[:, 0], self.height - obj[:, 1]]
        return np.c_[self._apply_scale(xy), np.full(len(xy), -self.glass_mm)]

    def window_poly(self, shrink: float = 0.0) -> np.ndarray:
        x0, y0, x1, y1 = self.window
        pts = [(x0 + shrink, y0 + shrink), (x1 - shrink, y0 + shrink),
               (x1 - shrink, y1 - shrink), (x0 + shrink, y1 - shrink)]
        return self._apply_scale(pts)

    def texture(self, px_per_square: int) -> np.ndarray:
        """Board image (row 0 = top = max y) with the middle blanked to white."""
        img = self.board().generateImage((self.squares_x * px_per_square, self.squares_y * px_per_square),
                                         marginSize=0)
        x0, y0, x1, y1 = (int(v * px_per_square) for v in self.blank_sq)
        top = self.squares_y * px_per_square
        img[top - y1:top - y0, x0:x1] = 255
        return img

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, name_or_path) -> "Sheet":
        p = Path(name_or_path)
        if not p.suffix:
            p = config.sheets_dir() / str(name_or_path) / "sheet.json"
        if not p.exists():
            raise FileNotFoundError(f"altlık bulunamadı: {name_or_path} (önce make_reference_sheet)")
        return cls(**json.loads(p.read_text(encoding="utf-8")))


def detect(gray, sheet: Sheet):
    """ChArUco corners found in the image: (corners (N,2), ids (N,))."""
    det = cv2.aruco.CharucoDetector(sheet.board())
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None:
        return np.zeros((0, 2)), np.zeros(0, int)
    return cc.reshape(-1, 2).astype(np.float64), ci.ravel().astype(int)


def build(sheet: Sheet, out_dir: Path) -> dict:
    """A4 PDF at exact scale (print at 100 %), a preview PNG and sheet.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = DPI / 25.4
    sq = int(round(sheet.square_mm * mm))
    if abs(sq / mm - sheet.square_mm) > 1e-6:
        raise ValueError(f"kare boyu {DPI} dpi'da tam piksel olmalı (ör. 12.7 mm)")
    tex = sheet.texture(sq)
    pw, ph = int(round(PAGE_MM[0] * mm)), int(round(PAGE_MM[1] * mm))
    if tex.shape[1] > pw or tex.shape[0] > ph:
        raise ValueError("desen A4'e sığmıyor")
    page = np.full((ph, pw), 255, np.uint8)
    ox, oy = (pw - tex.shape[1]) // 2, (ph - tex.shape[0]) // 2
    page[oy:oy + tex.shape[0], ox:ox + tex.shape[1]] = tex
    im = Image.fromarray(page)
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, int(2.6 * mm))
    small = ImageFont.truetype(FONT, int(2.2 * mm))

    def to_px(x, y):                         # sheet mm (y up) -> page px
        return ox + x * mm, oy + (sheet.height - y) * mm

    r = sheet.ref_points
    bottom = oy + tex.shape[0]
    left = ox
    for name, (x, y) in r.items():
        px, py = to_px(x, y)
        if name in ("A", "B"):              # arrow in the bottom margin pointing up at the corner column
            tip = bottom + 0.8 * mm
            dr.polygon([(px, tip), (px - 1.4 * mm, tip + 2.4 * mm), (px + 1.4 * mm, tip + 2.4 * mm)], fill=0)
            dr.text((px + 1.8 * mm, tip + 0.2 * mm), name, font=font, fill=0)
        if name in ("A", "C"):              # arrow in the left margin pointing right at the corner row
            tip = left - 0.8 * mm
            dr.polygon([(tip, py), (tip - 2.4 * mm, py - 1.4 * mm), (tip - 2.4 * mm, py + 1.4 * mm)], fill=0)
            dr.text((tip - 2.6 * mm, py - 5.2 * mm), name, font=font, fill=0)
    note = (f"foto-olcum altlık '{sheet.name}'  |  %100 / gerçek boyutta basın  |  "
            f"A-B ve A-C: {REF_SQUARES} kare = {sheet.ref_nominal:.1f} mm (kare köşesinden köşesine kumpasla ölçün)")
    dr.text((ox, oy - 4.2 * mm), note, font=small, fill=0)

    files = {"pdf": out_dir / f"{sheet.name}.pdf", "preview_png": out_dir / f"{sheet.name}_onizleme.png",
             "config": out_dir / "sheet.json"}
    im.convert("1").save(files["pdf"], resolution=DPI)
    im.resize((pw // 6, ph // 6), Image.BOX).save(files["preview_png"])
    sheet.save(files["config"])
    return {k: str(v) for k, v in files.items()}


def read_edid(connector: str) -> dict:
    """Preferred resolution and physical size (mm) of a connected display from its EDID."""
    hits = sorted(Path("/sys/class/drm").glob(f"card*-{connector}"))
    if not hits:
        names = sorted(p.name.split("-", 1)[1] for p in Path("/sys/class/drm").glob("card*-*"))
        raise FileNotFoundError(f"ekran bulunamadı: {connector}; mevcut: {names}")
    e = (hits[0] / "edid").read_bytes()
    if len(e) < 128:
        raise ValueError(f"{connector}: EDID okunamadı (ekran bağlı mı?)")
    d = e[54:72]
    w_px, h_px = d[2] | ((d[4] & 0xF0) << 4), d[5] | ((d[7] & 0xF0) << 4)
    w_mm, h_mm = d[12] | ((d[14] & 0xF0) << 4), d[13] | ((d[14] & 0x0F) << 8)
    status = (hits[0] / "status").read_text().strip()
    enabled = (hits[0] / "enabled").read_text().strip() if (hits[0] / "enabled").exists() else "?"
    return {"connector": connector, "px": [w_px, h_px], "mm": [w_mm, h_mm], "status": status, "enabled": enabled}


def screen_layout(name: str, edid: dict, square_px: int = 64, border: int = 2,
                  glass_mm: float = 0.8, glass_sigma_mm: float = 0.4) -> Sheet:
    """ChArUco filling a screen, blank middle, A-B / A-C as long as a 150 mm caliper allows."""
    (w_px, h_px), (w_mm, h_mm) = edid["px"], edid["mm"]
    pitch = w_mm / w_px                     # the larger dimension has the smaller EDID rounding error
    nx, ny = w_px // square_px, h_px // square_px
    sq = square_px * pitch
    n_ref = int(min(140.0 // sq, nx - 2, ny - 2))
    dict_name = next(d for d, n in (("DICT_4X4_250", 250), ("DICT_4X4_1000", 1000)) if nx * ny // 2 + 1 <= n)
    return Sheet(name, nx, ny, sq, 0.75 * sq, border, None, dict_name, "screen", None, [1, 1], n_ref,
                 glass_mm, glass_sigma_mm, square_px, edid)


def build_screen(sheet: Sheet, out_dir: Path) -> dict:
    """Full-screen PNGs at native resolution: one to photograph, one with A/B/C rings to
    measure with the caliper. Both have identical pixel geometry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w_px, h_px = sheet.screen["px"]
    sq = sheet.square_px
    tex = sheet.texture(sq)
    page = np.full((h_px, w_px), 255, np.uint8)
    ox, oy = (w_px - tex.shape[1]) // 2, (h_px - tex.shape[0]) // 2
    page[oy:oy + tex.shape[0], ox:ox + tex.shape[1]] = tex
    shoot = Image.fromarray(page).convert("RGB")
    meas = shoot.copy()
    dr = ImageDraw.Draw(meas)
    font = ImageFont.truetype(FONT, 28)
    for name, (x, y) in sheet.ref_points.items():
        px, py = ox + x / sheet.square_mm * sq, oy + (sheet.squares_y - y / sheet.square_mm) * sq
        r = sq * 0.35
        dr.ellipse([px - r, py - r, px + r, py + r], outline=(230, 0, 0), width=4)
        dr.text((px + r + 4, py - r - 34), name, font=font, fill=(230, 0, 0))
    x0, y0, x1, y1 = sheet.blank_sq
    tx, ty = ox + (x0 + 0.5) * sq, oy + (sheet.squares_y - y1 + 0.5) * sq
    dr.multiline_text((tx, ty), (
        f"foto-olcum ekran altlığı '{sheet.name}'\n"
        f"Kırmızı halkalı dama köşeleri: A-B ve A-C = {sheet.ref_squares} kare = {sheet.ref_nominal:.2f} mm (nominal)\n"
        "Kumpas uçlarını köşelere koyup ölçün, set_sheet_measurement ile girin.\n"
        "Fotoğraf için ekran_cekim.png'yi açın (halkalar olmadan)."), font=font, fill=(200, 0, 0), spacing=10)
    files = {"shoot_png": out_dir / "ekran_cekim.png", "measure_png": out_dir / "ekran_olcum.png",
             "config": out_dir / "sheet.json"}
    shoot.save(files["shoot_png"])
    meas.save(files["measure_png"])
    sheet.save(files["config"])
    return {k: str(v) for k, v in files.items()}
