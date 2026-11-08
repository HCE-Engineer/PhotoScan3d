"""Runs inside FreeCAD (freecadcmd). Do not import from foto_olcum here.

Job file path comes from env FOTO_OLCUM_JOB:
  {"name": str, "out_dir": str,
   "measurements": {id: {"value_mm": float, "uncertainty_mm": float, "source": str,
                         "locked": bool, "label": str}},
   "model": {"features": [...]}}
Writes <out_dir>/result.json with {"ok": bool, ...}. A missing result.json means failure.
"""
import json
import math
import os
import re
import traceback

import FreeCAD as App
import Part
import Sketcher

V = App.Vector
TOL = 1e-4


class BuildError(Exception):
    pass


class Ctx:
    def __init__(self, job):
        self.job = job
        self.meas = job["measurements"]
        self.values = {k: float(v["value_mm"]) for k, v in self.meas.items()}
        names = sorted(self.values, key=len, reverse=True)
        self._name_re = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, names)) + r")(?![\w])") if names else None
        self.features = {}
        self.sketch_report = []

    def expr(self, s):
        """'plate_w - 5' -> 'Params.plate_w - 5'"""
        s = str(s)
        return self._name_re.sub(r"Params.\1", s) if self._name_re else s

    def val(self, s):
        if isinstance(s, (int, float)):
            return float(s)
        env = {"__builtins__": {}, "pi": math.pi, "sqrt": math.sqrt, "sin": math.sin,
               "cos": math.cos, "tan": math.tan}
        try:
            return float(eval(str(s), env, dict(self.values)))
        except Exception as e:
            raise BuildError(f"ifade hesaplanamadı: {s!r} ({e})")

    def bind(self, obj, prop, s):
        if isinstance(s, (int, float)):
            return
        obj.setExpression(prop, self.expr(s))


def make_params(doc, ctx):
    sh = doc.addObject("Spreadsheet::Sheet", "Params")
    for col, title in zip("ABCDEF", ["id", "value_mm", "uncertainty_mm", "source", "locked", "label"]):
        sh.set(f"{col}1", title)
    for row, (mid, m) in enumerate(sorted(ctx.meas.items()), start=2):
        sh.set(f"A{row}", mid)
        sh.set(f"B{row}", repr(float(m["value_mm"])))
        sh.setAlias(f"B{row}", mid)
        sh.set(f"C{row}", repr(float(m.get("uncertainty_mm", 0.0))))
        sh.set(f"D{row}", str(m.get("source", "")))
        sh.set(f"E{row}", "1" if m.get("locked") else "0")
        sh.set(f"F{row}", "'" + str(m.get("label", "")))
    return sh


def plane_of(body, name):
    want = {"XY": "XY_Plane", "XZ": "XZ_Plane", "YZ": "YZ_Plane"}[name]
    for f in body.Origin.OriginFeatures:
        if getattr(f, "Role", None) == want or f.Name.startswith(want):
            return f
    raise BuildError(f"düzlem bulunamadı: {name}")


def new_sketch(doc, body, ctx, fid, plane, z=None):
    sk = body.newObject("Sketcher::SketchObject", f"{fid}_sketch")
    sk.AttachmentSupport = (plane_of(body, plane), [""])
    sk.MapMode = "FlatFace"
    if z is not None:
        sk.AttachmentOffset = App.Placement(V(0, 0, ctx.val(z)), App.Rotation())
        ctx.bind(sk, ".AttachmentOffset.Base.z", z)
    return sk


def add_bound(sk, ctx, cons, expr_):
    i = sk.addConstraint(cons)
    if not isinstance(expr_, (int, float)):
        sk.setExpression(f"Constraints[{i}]", ctx.expr(expr_))
    return i


def _corner(prev, v, nxt, r):
    """Tangent points, center and turn sign of a fillet of radius r at corner v."""
    ax, ay = v[0] - prev[0], v[1] - prev[1]
    bx, by = nxt[0] - v[0], nxt[1] - v[1]
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    ua, ub = (ax / la, ay / la), (bx / lb, by / lb)
    cross = ua[0] * ub[1] - ua[1] * ub[0]
    turn = math.atan2(cross, ua[0] * ub[0] + ua[1] * ub[1])
    if abs(turn) < 1e-6:
        raise BuildError(f"({v[0]:.2f}, {v[1]:.2f}) köşesi düz, radyus verilemez")
    d = r * math.tan(abs(turn) / 2)
    ta = (v[0] - d * ua[0], v[1] - d * ua[1])
    tb = (v[0] + d * ub[0], v[1] + d * ub[1])
    s = 1 if turn > 0 else -1          # +1: left turn, center on the left of both edges
    c = (ta[0] - s * r * ua[1], ta[1] + s * r * ua[0])
    return ta, tb, c, s, d


def polygon(sk, ctx, pts):
    """Closed profile. Each vertex is [x, y] or [x, y, r] (r: corner radius, tangent arc).

    The sharp polygon is drawn as construction lines whose vertices are bound to the
    sheet; real edges lie on them and fillet arcs are tangent to both neighbours, so the
    sketch stays fully constrained and has no topological-naming dependency."""
    n = len(pts)
    if n < 3:
        raise BuildError("profil en az 3 köşe olmalı")
    xy = [(ctx.val(p[0]), ctx.val(p[1])) for p in pts]
    radii = [p[2] if len(p) > 2 and p[2] not in (None, 0, "0") else None for p in pts]
    rv = [ctx.val(r) if r is not None else 0.0 for r in radii]

    first = sk.GeometryCount
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        sk.addGeometry(Part.LineSegment(V(a[0], a[1], 0), V(b[0], b[1], 0)), True)
    for i in range(n):
        sk.addConstraint(Sketcher.Constraint("Coincident", first + i, 2, first + (i + 1) % n, 1))
    for i, p in enumerate(pts):
        add_bound(sk, ctx, Sketcher.Constraint("DistanceX", -1, 1, first + i, 1, xy[i][0]), p[0])
        add_bound(sk, ctx, Sketcher.Constraint("DistanceY", -1, 1, first + i, 1, xy[i][1]), p[1])

    corners = {}
    for i in range(n):
        if radii[i] is not None:
            corners[i] = _corner(xy[i - 1], xy[i], xy[(i + 1) % n], rv[i])
    # real edge i runs from vertex i to vertex i+1, shortened by fillets
    lines = []
    for i in range(n):
        j = (i + 1) % n
        a = corners[i][1] if i in corners else xy[i]
        b = corners[j][0] if j in corners else xy[j]
        used = (corners[i][4] if i in corners else 0) + (corners[j][4] if j in corners else 0)
        if used >= math.hypot(xy[j][0] - xy[i][0], xy[j][1] - xy[i][1]) - 1e-6:
            raise BuildError(f"{i}-{j} kenarı radyuslar için çok kısa")
        lines.append(sk.addGeometry(Part.LineSegment(V(a[0], a[1], 0), V(b[0], b[1], 0))))
    for i in range(n):
        j = (i + 1) % n
        for end, vtx in ((1, i), (2, j)):
            if vtx in corners:
                sk.addConstraint(Sketcher.Constraint("PointOnObject", lines[i], end, first + i))
            else:
                cons_end = 1 if vtx == i else 2
                sk.addConstraint(Sketcher.Constraint("Coincident", lines[i], end, first + i, cons_end))
    for i, (ta, tb, c, s, _) in corners.items():
        a0 = math.atan2(ta[1] - c[1], ta[0] - c[0])
        a1 = math.atan2(tb[1] - c[1], tb[0] - c[0])
        if s < 0:
            a0, a1 = a1, a0
        while a1 <= a0:
            a1 += 2 * math.pi
        arc = sk.addGeometry(Part.ArcOfCircle(Part.Circle(V(c[0], c[1], 0), V(0, 0, 1), rv[i]), a0, a1))
        prev_line, next_line = lines[i - 1], lines[i]
        # arcs run counter-clockwise: a left turn starts at the incoming edge
        start_line, start_end, end_line, end_end = (prev_line, 2, next_line, 1) if s > 0 else (next_line, 1, prev_line, 2)
        sk.addConstraint(Sketcher.Constraint("Tangent", arc, 1, start_line, start_end))
        sk.addConstraint(Sketcher.Constraint("Tangent", arc, 2, end_line, end_end))
        add_bound(sk, ctx, Sketcher.Constraint("Radius", arc, rv[i]), radii[i])
    return xy


def circle(sk, ctx, center, dia):
    cx, cy, d = ctx.val(center[0]), ctx.val(center[1]), ctx.val(dia)
    g = sk.addGeometry(Part.Circle(V(cx, cy, 0), V(0, 0, 1), d / 2))
    add_bound(sk, ctx, Sketcher.Constraint("DistanceX", -1, 1, g, 3, cx), center[0])
    add_bound(sk, ctx, Sketcher.Constraint("DistanceY", -1, 1, g, 3, cy), center[1])
    add_bound(sk, ctx, Sketcher.Constraint("Diameter", g, d), dia)


def check_sketch(sk, ctx):
    sk.recompute()
    dof = sk.solve()
    rep = {"sketch": sk.Name, "solve": dof,
           "fully_constrained": bool(getattr(sk, "FullyConstrained", False)),
           "conflicting": list(getattr(sk, "Conflicting", []) or []),
           "redundant": list(getattr(sk, "Redundant", []) or [])}
    ctx.sketch_report.append(rep)
    if rep["conflicting"] or rep["redundant"]:
        raise BuildError(f"{sk.Name}: çelişen/fazla kısıt {rep}")


def top_z(ctx):
    for f in ctx.job["model"]["features"]:
        if f["type"] == "pad":
            return f.get("z", 0), f["length"]
    raise BuildError("önce bir pad gerekli")


def feat_pad(doc, body, ctx, f):
    sk = new_sketch(doc, body, ctx, f["id"], f.get("plane", "XY"), f.get("z"))
    if "circle" in f:
        circle(sk, ctx, f["circle"]["center"], f["circle"]["diameter"])
    else:
        polygon(sk, ctx, f["profile"])
    for h in f.get("holes", []):
        circle(sk, ctx, h["center"], h["diameter"])
    check_sketch(sk, ctx)
    pad = body.newObject("PartDesign::Pad", f["id"])
    pad.Profile = sk
    pad.Length = ctx.val(f["length"])
    ctx.bind(pad, "Length", f["length"])
    return pad


def _cut_top(ctx, f):
    if "z" in f:
        return f["z"]
    z0, length = top_z(ctx)
    return f"({z0}) + ({length})" if not isinstance(z0, (int, float)) or z0 else length


def feat_pocket(doc, body, ctx, f, profile_fn):
    sk = new_sketch(doc, body, ctx, f["id"], f.get("plane", "XY"), _cut_top(ctx, f))
    profile_fn(sk)
    check_sketch(sk, ctx)
    p = body.newObject("PartDesign::Pocket", f["id"])
    p.Profile = sk
    depth = f.get("depth", "through")
    if depth == "through":
        p.Type = "ThroughAll"
    else:
        p.Type = "Length"
        p.Length = ctx.val(depth)
        ctx.bind(p, "Length", depth)
    return p


def build(job):
    ctx = Ctx(job)
    doc = App.newDocument(re.sub(r"\W", "_", job["name"]) or "part")
    make_params(doc, ctx)
    body = doc.addObject("PartDesign::Body", "Body")
    doc.recompute()
    for f in job["model"]["features"]:
        t = f["type"]
        if t == "pad":
            o = feat_pad(doc, body, ctx, f)
        elif t == "pocket":
            o = feat_pocket(doc, body, ctx, f, lambda sk, f=f: polygon(sk, ctx, f["profile"]))
        elif t == "hole":
            o = feat_pocket(doc, body, ctx, f, lambda sk, f=f: circle(sk, ctx, f["center"], f["diameter"]))
            if f.get("thread"):
                o.Label2 = f"thread {f['thread']}"
        else:
            raise BuildError(f"desteklenmeyen feature tipi: {t}")
        doc.recompute()
        if o.Shape.isNull() or not o.Shape.isValid():
            raise BuildError(f"{f['id']} ({t}) geçersiz şekil üretti")
        ctx.features[f["id"]] = o.Name

    doc.recompute()
    shape = body.Shape
    if shape.isNull() or not shape.isValid():
        raise BuildError("gövde şekli geçersiz")
    bad = [o.Name for o in doc.Objects if "Invalid" in o.State or "Error" in o.State]
    if bad:
        raise BuildError(f"recompute hatası: {bad}")

    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, job["name"])
    doc.saveAs(base + ".FCStd")
    shape.exportStep(base + ".step")
    import MeshPart
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01, AngularDeflection=0.087)
    mesh.write(base + ".stl")
    bb = shape.BoundBox
    return {
        "fcstd": base + ".FCStd", "step": base + ".step", "stl": base + ".stl",
        "volume_mm3": shape.Volume,
        "bbox_mm": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
        "sketches": ctx.sketch_report,
        "all_fully_constrained": all(r["fully_constrained"] for r in ctx.sketch_report),
    }


def main():
    job_path = os.environ.get("FOTO_OLCUM_JOB")
    res_path = os.path.join(os.path.dirname(job_path or "."), "result.json")
    try:
        with open(job_path, encoding="utf-8") as fh:
            job = json.load(fh)
        res_path = os.path.join(job["out_dir"], "result.json")
        os.makedirs(job["out_dir"], exist_ok=True)
        res = {"ok": True, **build(job)}
    except BuildError as e:
        res = {"ok": False, "error": str(e)}
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    with open(res_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


main()
