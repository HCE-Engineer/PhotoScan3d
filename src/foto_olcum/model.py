"""Feature tree (model.json) validation and expression evaluation."""
import ast
import math

# Faz 1: pad, pocket, hole. Later: slot, chamfer, pattern_*, revolve, loft.
SUPPORTED = {"pad", "pocket", "hole"}
PLANNED = {"slot", "chamfer", "fillet", "pattern_linear", "pattern_polar", "revolve", "loft"}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi}


def names_in(expr) -> set[str]:
    if isinstance(expr, (int, float)):
        return set()
    tree = ast.parse(str(expr), mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(_FUNCS)


def evaluate(expr, values: dict[str, float]) -> float:
    if isinstance(expr, (int, float)):
        return float(expr)
    tree = ast.parse(str(expr), mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"ifadede izin verilmeyen öğe: {expr!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError(f"ifadede sadece {sorted(k for k in _FUNCS if k != 'pi')} çağrılabilir: {expr!r}")
        if isinstance(node, ast.Name) and node.id not in _FUNCS and node.id not in values:
            raise ValueError(f"tanımsız ölçü '{node.id}' ({expr!r})")
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {**_FUNCS, **values}))


def _numbers(f: dict) -> list:
    out = []
    t = f["type"]
    if t in ("pad", "pocket"):
        for v in f["profile"]:
            out += list(v)
        for h in f.get("holes", []):
            out += [*h["center"], h["diameter"]]
    if t == "hole":
        out += [*f["center"], f["diameter"]]
    for k in ("length", "z"):
        if k in f:
            out.append(f[k])
    if f.get("depth", "through") != "through":
        out.append(f["depth"])
    return [x for x in out if x is not None]


def validate(model: dict, values: dict[str, float]) -> list[str]:
    """Return a list of human-readable problems (empty = ok)."""
    errs = []
    feats = model.get("features")
    if not isinstance(feats, list) or not feats:
        return ["model.features boş olmamalı bir liste olmalı"]
    if feats[0].get("type") != "pad":
        errs.append("ilk feature bir pad olmalı (taban gövde)")
    seen = set()
    for i, f in enumerate(feats):
        fid, t = f.get("id"), f.get("type")
        where = f"features[{i}] ({fid})"
        if not fid or not str(fid).isidentifier():
            errs.append(f"{where}: geçerli bir id gerekli")
        if fid in seen:
            errs.append(f"{where}: id tekrar ediyor")
        seen.add(fid)
        if t in PLANNED:
            errs.append(f"{where}: '{t}' henüz desteklenmiyor (planlı). Dik köşe radyusu için "
                        "profil köşesine üçüncü eleman olarak yarıçap verin: [x, y, r]")
            continue
        if t not in SUPPORTED:
            errs.append(f"{where}: bilinmeyen tip '{t}', desteklenen: {sorted(SUPPORTED)}")
            continue
        if f.get("plane", "XY") not in ("XY", "XZ", "YZ"):
            errs.append(f"{where}: plane XY/XZ/YZ olmalı")
        req = {"pad": ["profile", "length"], "pocket": ["profile"], "hole": ["center", "diameter"]}[t]
        missing = [k for k in req if k not in f]
        if missing:
            errs.append(f"{where}: eksik alan {missing}")
            continue
        if "profile" in f:
            prof = f["profile"]
            if len(prof) < 3 or any(len(v) not in (2, 3) for v in prof):
                errs.append(f"{where}: profile en az 3 köşe, her köşe [x, y] veya [x, y, r]")
                continue
        for expr in _numbers(f):
            try:
                unknown = names_in(expr) - set(values)
                if unknown:
                    errs.append(f"{where}: tanımsız ölçü {sorted(unknown)} ('{expr}')")
                else:
                    evaluate(expr, values)
            except (SyntaxError, ValueError, ZeroDivisionError) as e:
                errs.append(f"{where}: ifade hatası '{expr}': {e}")
    return errs
