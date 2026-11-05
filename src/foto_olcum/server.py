"""MCP server. Never print to stdout here: stdio is the protocol channel."""
import logging
import sys
from dataclasses import asdict

from mcp.server.mcpserver import MCPServer

from . import build as build_mod
from .model import validate
from .project import LockedError, Measurement, Project

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("foto-olcum")

INSTRUCTIONS = """\
foto-olcum: prizmatik parçaların fotoğraflarından parametrik FreeCAD modeli.
Sen şekli yorumlar ve feature ağacını kurarsın; sayıları bu araçlardan alırsın, pikselden
kendin ölçü tahmin etmezsin. Her ölçünün belirsizliği vardır; kumpasla girilenler kilitlidir.
Akış: new_project → add_photos → process_photos → extract_features_2d → set_model →
build_freecad → render_and_compare → kritik ölçüler için kumpas → set_measurement → tekrar.
Kullanıcıya Türkçe yaz."""

mcp = MCPServer("foto-olcum", instructions=INSTRUCTIONS)


def _err(e: Exception) -> dict:
    return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _table(p: Project) -> list[dict]:
    return [{"id": m.id, "label": m.label, "value_mm": round(m.value_mm, 3),
             "uncertainty_mm": round(m.uncertainty_mm, 3), "source": m.source,
             "locked": m.locked, "role": m.role, "material": m.material}
            for m in sorted(p.measurements().values(), key=lambda m: m.id)]


@mcp.tool()
def new_project(name: str) -> dict:
    """Yeni parça projesi oluşturur. name: klasör adı (harf, rakam, _ ve -)."""
    try:
        p = Project.create(name)
        return {"ok": True, "project": name, "path": str(p.dir)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def status(project: str) -> dict:
    """Ölçü tablosu (değer, ±, kaynak, kilit), fotoğraflar, son build ve compare sonucu."""
    try:
        p = Project.open(project)
        ms = p.measurements()
        st = p.state()
        return {"ok": True, "path": str(p.dir), "measurements": _table(p),
                "photos": p.photos(), "has_model": p.model() is not None,
                "locked": sorted(k for k, m in ms.items() if m.locked),
                "last_build": st.get("last_build"), "last_compare": st.get("last_compare_summary")}
    except Exception as e:
        return _err(e)


@mcp.tool()
def set_measurement(project: str, id: str, value_mm: float, label: str = "",
                    source: str = "caliper", uncertainty_mm: float | None = None,
                    role: str = "", material: str = "", replace_caliper: bool = False) -> dict:
    """Ölçü girer. source='caliper' ise ölçü kilitlenir (varsayılan belirsizlik 0.03 mm)
    ve fotoğraf analizi onu bir daha değiştirmez. role: 'critical' | 'control' | ''.
    Kontrol ölçüleri kalibrasyonda kullanılmaz, sadece doğrulamada. Kilitli bir kumpas
    değerini düzeltmek için replace_caliper=True."""
    try:
        p = Project.open(project)
        old = p.measurements().get(id)
        if uncertainty_mm is None:
            uncertainty_mm = 0.03 if source == "caliper" else (old.uncertainty_mm if old else 0.5)
        m = Measurement(id=id, value_mm=float(value_mm), uncertainty_mm=float(uncertainty_mm),
                        source=source, label=label or (old.label if old else ""),
                        locked=(source == "caliper"), role=role or (old.role if old else ""),
                        material=material or (old.material if old else ""),
                        feature_ref=old.feature_ref if old else "",
                        evidence=old.evidence if old else [])
        p.put_measurement(m, force=replace_caliper)
        out = {"ok": True, "measurement": asdict(m)}
        if old:
            out["previous"] = {"value_mm": old.value_mm, "uncertainty_mm": old.uncertainty_mm,
                               "source": old.source}
            out["delta_mm"] = round(m.value_mm - old.value_mm, 4)
            if not old.locked and old.uncertainty_mm > 0:
                out["delta_in_sigma"] = round(abs(out["delta_mm"]) / old.uncertainty_mm, 2)
        return out
    except LockedError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def set_model(project: str, model: dict) -> dict:
    """Feature ağacını (model.json) kaydeder ve doğrular. Sayılar ölçü id'si, ifade
    ('plate_w - 5') veya sabit olabilir. Faz 1 tipleri:
      pad:    {id, type:'pad', plane:'XY', profile:[[x,y] veya [x,y,r], ...], length, z?}
      pocket: {id, type:'pocket', profile:[...], depth:'through'|ifade, z?}
      hole:   {id, type:'hole', center:[x,y], diameter, depth:'through'|ifade, thread?:'M6'}
    Profil köşesindeki r, o dik köşenin radyusudur (sketch içinde teğet yay).
    Koordinatlar parça sisteminde: XY oturma yüzeyi (çerçeve düzlemi), Z yukarı."""
    try:
        p = Project.open(project)
        values = {k: m.value_mm for k, m in p.measurements().items()}
        errs = validate(model, values)
        if errs:
            return {"ok": False, "problems": errs}
        p.set_model(model)
        return {"ok": True, "n_features": len(model["features"])}
    except Exception as e:
        return _err(e)


@mcp.tool()
def build_freecad(project: str) -> dict:
    """Modeli FreeCAD'de üretir: Params tablosu (alias = ölçü id), tam kısıtlı sketch'ler,
    .FCStd + .step + .stl. Hata varsa FreeCAD log'unun sonu döner."""
    try:
        return build_mod.build_project(Project.open(project))
    except Exception as e:
        return _err(e)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
