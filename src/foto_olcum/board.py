"""Printed ChArUco base sheet (A4) with a blank middle area for the part.

Sheet coordinates (mm): origin at the board's lower-left corner, x right, y up, z up
(toward the camera); the sheet is the plane z = 0. OpenCV's ChArUco object points run
x right, y *down* from the top-left, so y_sheet = height - y_board.

On a white backlight the paper glows, printed squares stay dark, and the part in the blank
middle shows as a silhouette. ChArUco chessboard corners are located far more precisely
than single ArUco marker corners. The same sheet also serves for camera calibration.

Printers scale by ~0.5-1 %; the user measures 10 squares along x and y with a caliper
(marked A-B and A-C on the sheet) and the object points are scaled accordingly.
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
    border: int = 2               # squares of pattern around the blank middle
    measured_ref: list | None = None   # [A-B, A-C] in mm, by caliper
    dictionary: str = DICT_NAME

    @property
    def width(self):
        return self.squares_x * self.square_mm

    @property
    def height(self):
        return self.squares_y * self.square_mm

    @property
    def window(self):
        """[x0, y0, x1, y1] of the blank middle area (nominal sheet mm)."""
        b = self.border * self.square_mm
        return [b, b, self.width - b, self.height - b]

    @property
    def ref_nominal(self):
        return REF_SQUARES * self.square_mm

    @property
    def ref_points(self):
        """Caliper points A, B (= A + 10 squares in x), C (= A + 10 squares in y)."""
        a = np.array([1.0, 1.0]) * self.square_mm
        d = self.ref_nominal
        return {"A": a, "B": a + [d, 0], "C": a + [0, d]}

    @property
    def scale(self):
        if not self.measured_ref:
            return 1.0, 1.0
        return self.measured_ref[0] / self.ref_nominal, self.measured_ref[1] / self.ref_nominal

    @property
    def scale_sigma_rel(self):
        """Printer scale error if unmeasured, else caliper reading on printed corners."""
        if not self.measured_ref:
            return 0.01
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
        return np.c_[self._apply_scale(xy), np.zeros(len(xy))]

    def window_poly(self, shrink: float = 0.0) -> np.ndarray:
        x0, y0, x1, y1 = self.window
        pts = [(x0 + shrink, y0 + shrink), (x1 - shrink, y0 + shrink),
               (x1 - shrink, y1 - shrink), (x0 + shrink, y1 - shrink)]
        return self._apply_scale(pts)

    def texture(self, px_per_square: int) -> np.ndarray:
        """Board image (row 0 = top = max y) with the middle blanked to white."""
        img = self.board().generateImage((self.squares_x * px_per_square, self.squares_y * px_per_square),
                                         marginSize=0)
        b = self.border * px_per_square
        img[b:img.shape[0] - b, b:img.shape[1] - b] = 255
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
