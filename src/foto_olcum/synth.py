"""Synthetic backlit photos of a known 2.5D part on the printed ChArUco sheet.

Deliberately independent of the measurement code path: its own pinhole projection
(no cv2.projectPoints / backproject / ChArUco detection), PIL polygon rasterisation
with supersampling, the sheet texture warped through its own projected corners,
then blur, noise and JPEG. Geometry bugs on either side therefore do not cancel.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .board import Sheet


@dataclass
class Part:
    outline: list | None = None       # [(x, y, r)] CCW in part coords, r = corner radius
    holes: list = field(default_factory=list)    # [(cx, cy, d)]
    thickness: float = 5.0
    place: tuple = (60.0, 70.0, 10.0)            # x, y, rotation_deg in sheet coords
    outer_dia: float = 0.0            # round part (disc, ring, bearing) centred on the part origin


@dataclass
class Cam:
    f: float = 2600.0
    w: int = 3000
    h: int = 2250
    dist_mm: float = 380.0
    tilt_deg: tuple = (4.0, -3.0)     # rotation about x, y
    target: tuple = (95.25, 139.7)    # sheet point the camera looks at (A4 sheet centre)

    def K(self):
        return np.array([[self.f, 0, (self.w - 1) / 2], [0, self.f, (self.h - 1) / 2], [0, 0, 1.0]])

    def Rt(self):
        """World (sheet) -> camera. Camera looks down -z of the world (z up)."""
        ax, ay = map(math.radians, self.tilt_deg)
        # base: camera x = world x, camera y = -world y, camera z = -world z
        base = np.diag([1.0, -1.0, -1.0])
        rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        R = ry @ rx @ base
        target = np.array([*self.target, 0.0])
        C = target - R.T @ np.array([0, 0, self.dist_mm])     # camera center in world
        t = -R @ C
        return R, t


def _proj(cam: Cam, pts3):
    R, t = cam.Rt()
    pc = (R @ np.asarray(pts3, float).T).T + t
    K = cam.K()
    return np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]


def outline_points(part: Part, step_deg: float = 2.0):
    """Dense CCW outline in part coordinates with tangent fillet arcs (or a circle)."""
    if part.outer_dia:
        return _circle(0.0, 0.0, part.outer_dia, 720)
    v = [(x, y) for x, y, _ in part.outline]
    out = []
    n = len(v)
    for i in range(n):
        p0, p1, p2 = np.array(v[i - 1]), np.array(v[i]), np.array(v[(i + 1) % n])
        r = part.outline[i][2]
        if not r:
            out.append(p1)
            continue
        ua, ub = (p1 - p0) / np.linalg.norm(p1 - p0), (p2 - p1) / np.linalg.norm(p2 - p1)
        turn = math.atan2(ua[0] * ub[1] - ua[1] * ub[0], ua @ ub)
        d = r * math.tan(abs(turn) / 2)
        ta = p1 - d * ua
        s = 1 if turn > 0 else -1
        c = ta + s * r * np.array([-ua[1], ua[0]])
        a0 = math.atan2(*(ta - c)[::-1])
        steps = max(2, int(abs(math.degrees(turn)) / step_deg))
        for k in range(steps + 1):
            a = a0 + s * abs(turn) * k / steps
            out.append(c + r * np.array([math.cos(a), math.sin(a)]))
    return np.array(out)


def _place(part: Part, xy):
    x0, y0, rot = part.place
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (R @ np.asarray(xy, float).T).T + [x0, y0]


def _circle(cx, cy, d, n=360):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[cx + d / 2 * np.cos(a), cy + d / 2 * np.sin(a)]


def _clip(subject, clipper):
    """Sutherland-Hodgman: subject polygon clipped by a convex CCW clipper."""
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i - 1], clipper[i]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        for j in range(len(inp)):
            p, q = inp[j - 1], inp[j]
            sp, sq = side(p), side(q)
            if sq >= 0:
                if sp < 0:
                    out.append(p + (q - p) * sp / (sp - sq))
                out.append(q)
            elif sp >= 0:
                out.append(p + (q - p) * sp / (sp - sq))
    return np.array(out)


def _fill(dr, q, val):
    """PIL fills the outline pixels too, i.e. dilates by half a pixel; offset inward first."""
    q = np.asarray(q, float)
    area = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) / 2
    if abs(area) < 1.0:
        return
    s = 1.0 if area > 0 else -1.0
    e = np.roll(q, -1, 0) - q
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    n_in = s * np.c_[-e[:, 1], e[:, 0]]                 # inward normal of edge i (q[i] -> q[i+1])
    n_prev = np.roll(n_in, 1, 0)
    bis = n_in + n_prev
    cos_half = np.clip(np.einsum("ij,ij->i", bis / (np.linalg.norm(bis, axis=1, keepdims=True) + 1e-12), n_in), 0.2, 1)
    bis /= np.linalg.norm(bis, axis=1, keepdims=True) + 1e-12
    r = q + bis * (0.5 / cos_half)[:, None]
    a2 = np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]) / 2
    if a2 * area <= 0:
        return
    dr.polygon([tuple(p) for p in r], fill=val)


def _ccw(poly):
    a = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if a > 0 else poly[::-1]


def render(part: Part, sheet: Sheet, cam: Cam, ss: int = 3, blur_px: float = 0.7, noise: float = 2.0,
           seed: int = 0, levels=(28, 238, 222, 22)) -> Image.Image:
    """levels: (part, screen, backlit paper, printed black) grey values."""
    dark, screen, paper, black = levels
    W, H = cam.w * ss, cam.h * ss

    # sheet: the board texture mapped onto z = 0 through a homography of its own corners
    pps = 100                                           # texture px per square
    tex = sheet.texture(pps).astype(np.float32)
    th, tw = tex.shape
    # texture pixel centres sit at (i + 0.5) / pps squares; row 0 is the top (y = height)
    src = np.float32([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]])
    z = -sheet.glass_mm                                 # screen pixels sit below the glass
    world = [(0, sheet.height, z), (sheet.width, sheet.height, z), (sheet.width, 0, z), (0, 0, z)]
    dst = (_proj(cam, world) * ss + (ss - 1) / 2).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    shade = np.round(black + (paper - black) * tex / 255.0).astype(np.uint8)
    canvas = cv2.warpPerspective(shade, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=screen)
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)

    def poly(pts3, val):
        _fill(dr, _proj(cam, pts3) * ss + (ss - 1) / 2, val)   # pixel centres of the supersampled grid

    # part: union of top face, bottom face and side walls; through holes are bright where
    # the projected top and bottom openings overlap
    t = part.thickness
    o = _place(part, outline_points(part))
    for z in (0.0, t):
        poly(np.c_[o, np.full(len(o), z)], dark)
    for i in range(len(o)):
        a, b = o[i - 1], o[i]
        poly([(*a, 0), (*b, 0), (*b, t), (*a, t)], dark)
    for cx, cy, d in part.holes:
        c = _place(part, _circle(cx, cy, d))
        top = _ccw(_proj(cam, np.c_[c, np.full(len(c), t)]))
        bot = _ccw(_proj(cam, np.c_[c, np.zeros(len(c))]))
        inter = _clip(top, bot)
        if len(inter) >= 3:
            _fill(dr, inter * ss + (ss - 1) / 2, paper)       # the blank paper shows through

    im = im.filter(ImageFilter.GaussianBlur(blur_px * ss))
    im = im.resize((cam.w, cam.h), Image.BOX)
    a = np.asarray(im, np.float32)
    a += np.random.default_rng(seed).normal(0, noise, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
