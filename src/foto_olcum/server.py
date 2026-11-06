"""MCP server. Never print to stdout here: stdio is the protocol channel."""
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
from mcp.server.mcpserver import Image, MCPServer

from . import build as build_mod
from . import compare as compare_mod
from . import config, features2d
from . import measure as measure_mod
from .board import Sheet
from .board import build as build_sheet
from .board import detect as detect_sheet
from .calib import Camera, calibrate, imread, make_pattern
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
    Koordinatlar parça sisteminde: XY oturma yüzeyi (altlık düzlemi), Z yukarı."""
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


def _img(path):
    return Image(path=path) if path and Path(path).exists() else None


def _with_images(result: dict, *paths):
    imgs = [i for i in (_img(p) for p in paths) if i]
    return [result, *imgs] if imgs else result


@mcp.tool()
def make_reference_sheet(name: str = "a4"):
    """A4'e basılacak ChArUco altlığını üretir (PDF, tam ölçek): kenarlarda ChArUco deseni,
    ortada parça için boş alan. Ölçek kontrolü için A-B ve A-C (10 kare = 127 mm) işaretli."""
    try:
        sheet = Sheet(name)
        files = build_sheet(sheet, config.sheets_dir() / name)
        x0, y0, x1, y1 = sheet.window
        return _with_images({"ok": True, "sheet": name, "files": files,
                             "blank_area_mm": [round(x1 - x0, 1), round(y1 - y0, 1)], "talimat": [
            "PDF'i A4'e %100 / 'Gerçek boyut' ile basın ('Sayfaya sığdır' KAPALI). Lazer yazıcı en iyisi; "
            "mürekkepte koyu ve keskin baskı seçin.",
            "Kağıdın kıvrılmamasını sağlayın. Monitör/tablet beyaz ekranın üstüne düz koyun.",
            f"A-B ve A-C mesafelerini (kare köşesinden kare köşesine, nominal {sheet.ref_nominal:.1f} mm) "
            "kumpasla ölçüp set_sheet_measurement ile girin; yazıcının ölçek hatası buradan düzeltilir.",
            "Aynı kağıt kamera kalibrasyonu için de kullanılır: calibrate_camera(sheet=...)."]},
            files["preview_png"])
    except Exception as e:
        return _err(e)


@mcp.tool()
def set_sheet_measurement(sheet: str, ab_mm: float, ac_mm: float) -> dict:
    """Basılı altlıkta kumpasla ölçülen A-B (x) ve A-C (y) mesafeleri. ChArUco köşe
    koordinatları buna göre ölçeklenir."""
    try:
        sh = Sheet.load(sheet)
        sh.measured_ref = [float(ab_mm), float(ac_mm)]
        sh.save(config.sheets_dir() / sheet / "sheet.json")
        sx, sy = sh.scale
        out = {"ok": True, "scale_x": sx, "scale_y": sy}
        if abs(sx - 1) > 0.015 or abs(sy - 1) > 0.015:
            out["warning"] = ("nominalden %1.5'ten fazla fark: 'sayfaya sığdır' ile mi basıldı, ya da ölçüm "
                              "yanlış köşelerden mi alındı? Gerekirse %100 ile yeniden basın.")
        return out
    except Exception as e:
        return _err(e)


@mcp.tool()
def make_calibration_pattern(screen_w_px: int = 1920, screen_h_px: int = 1080):
    """(İsteğe bağlı) Monitörde gösterilecek ChArUco deseni. Basılı A4 altlık varsa ona gerek
    yok: calibrate_camera(sheet=...) ile altlığın fotoğrafları kullanılır."""
    try:
        out = make_pattern(config.calibration_dir() / f"charuco_{screen_w_px}x{screen_h_px}.png",
                           screen_w_px, screen_h_px)
        return {"ok": True, "png": str(out), "talimat": [
            "PNG'yi monitörde tam ekran açın, ölçeklemeyi %100 bırakın.",
            "Ölçümde kullanacağınız zoom ile ekranı 15-20 kez, farklı açılardan çekin."]}
    except Exception as e:
        return _err(e)


@mcp.tool()
def calibrate_camera(phone_name: str, photo_paths: list[str] | None = None, folder: str | None = None,
                     sheet: str | None = None) -> dict:
    """Kamera iç parametrelerini (K, distorsiyon) hesaplar ve phone_name adıyla kaydeder.
    sheet verilirse basılı altlığın fotoğrafları, verilmezse monitör deseni beklenir.
    Çekim: ölçümdeki zoom ile 15-20 foto, farklı açılar (±30°), desen bazen kadraj
    köşelerine gelecek şekilde. Reprojeksiyon hatası > 0.5 px ise uyarır."""
    try:
        paths = [Path(x) for x in (photo_paths or [])]
        if folder:
            paths += sorted(q for q in Path(folder).expanduser().iterdir()
                            if q.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        if not paths:
            return {"ok": False, "error": "fotoğraf verilmedi (photo_paths veya folder)"}
        return {"ok": True, **calibrate(paths, phone_name, Sheet.load(sheet) if sheet else None)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def add_photos(project: str, paths: list[str], sheet: str = "a4", kind: str = "silhouette",
               camera: str | None = None) -> dict:
    """Fotoğrafları projeye kopyalar ve altlıktaki ChArUco köşelerini hızlıca sayar.
    kind: 'silhouette' (arkadan aydınlatma, Faz 1). camera: calibrate_camera adı; yoksa
    EXIF'ten tahmin (daha az hassas)."""
    try:
        if kind != "silhouette":
            return {"ok": False, "error": "Faz 1'de sadece 'silhouette' destekleniyor"}
        p = Project.open(project)
        sh = Sheet.load(sheet)
        if camera:
            Camera.load(camera)
        out = []
        for src in paths:
            src = Path(src).expanduser()
            pid = p.add_photo(src, kind)
            p.update_photo(pid, sheet=sheet, camera=camera)
            g = cv2.cvtColor(imread(src), cv2.COLOR_BGR2GRAY)
            _, ids = detect_sheet(g, sh)
            out.append({"id": pid, "file": src.name, "size": list(g.shape[::-1]),
                        "charuco_corners": len(ids), "ok": len(ids) >= 40})
        bad = [o["id"] for o in out if not o["ok"]]
        res = {"ok": True, "photos": out}
        if bad:
            res["warning"] = (f"az ChArUco köşesi bulunan fotolar: {bad} (kağıdın tamamı kadrajda, düz ve "
                              "arkadan aydınlatılmış olmalı)")
        return res
    except Exception as e:
        return _err(e)


@mcp.tool()
def process_photos(project: str, photo_ids: list[str] | None = None):
    """Her foto için: distorsiyon düzeltme, ChArUco köşelerinden poz, mm/px, silüet maskesi,
    debug görseli. Kalınlık (thickness) girilmişse mm/px üst yüz düzlemi içindir."""
    try:
        p = Project.open(project)
        ids = photo_ids or list(p.photos())
        res, prev = [], []
        for pid in ids:
            try:
                r = features2d.process_photo(p, pid)
                prev.append(r["debug"]["preview"])
            except Exception as e:
                r = {"photo": pid, "error": f"{type(e).__name__}: {e}"}
            res.append(r)
        return _with_images({"ok": all("error" not in r for r in res), "photos": res}, *prev[:2])
    except Exception as e:
        return _err(e)


@mcp.tool()
def extract_features_2d(project: str, photo_id: str, axis_line: int | None = None,
                        origin_corner: int | None = None, create_measurements: bool = True,
                        material: str = ""):
    """Silüetten 2D özellikler: dış kontur (doğrular L*, köşeler V* = sanal keskin köşe +
    radyus), delikler (H*: merkez, çap), dairesel olmayan boşluklar (K*). Hepsi parça
    koordinatlarında mm ve 1σ belirsizlikle. Parça X ekseni varsayılan olarak en uzun düz
    kenar, orijin onun başlangıç köşesi; axis_line/origin_corner ile değiştirilebilir.
    create_measurements=True: o_v*_x/_y/_r ve h*_x/_y/_dia ölçülerini yazar (kilitli olanlara
    dokunmaz) ve doğrudan set_model'e verilebilecek bir suggested_model döndürür.
    Önce 'thickness' ölçüsü (kumpas) girilmiş olmalı. Debug görseline mutlaka bak."""
    try:
        p = Project.open(project)
        r = features2d.extract(p, photo_id, axis_line, origin_corner, create_measurements, material)
        return _with_images({"ok": True, **r}, r["debug"]["preview"])
    except Exception as e:
        return _err(e)


@mcp.tool()
def measure(project: str, photo_id: str, a: str, b: str | None = None, store_as: str | None = None,
            label: str = "") -> dict:
    """extract_features_2d sonucundan ölçü: a, b = 'V3', 'H1', 'L2' gibi adlar.
    V-V / H-H / V-H: merkezler arası mesafe (+ dx, dy); L-V / L-H: doğruya dik mesafe;
    L-L: paralel kenarlar arası mesafe ve açı; tek H: çap; tek V: radyus.
    store_as verilirse sonuç photo_coarse ölçüsü olarak kaydedilir."""
    try:
        p = Project.open(project)
        r = measure_mod.measure(p, photo_id, a, b)
        if store_as and "value_mm" in r:
            m = Measurement(store_as, r["value_mm"], r["uncertainty_mm"], "photo_coarse",
                            label or (f"{a}-{b}" if b else a), evidence=[f"debug/{photo_id}_ozellikler.png"])
            p.put_measurement(m)
            r["stored_as"] = store_as
        return {"ok": True, **r}
    except LockedError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return _err(e)


@mcp.tool()
def render_and_compare(project: str):
    """Son build'in mesh'ini her fotoğrafın pozuyla izdüşürür ve gerçek silüetle karşılaştırır:
    IoU, maks/p95 sapma (mm), sapma bölgeleri → en yakın feature, fark görseli
    (kırmızı = modelde fazla, mavi = modelde eksik). Hedef: tüm fotolarda maks < 0.3 mm."""
    try:
        p = Project.open(project)
        r = compare_mod.compare_project(p)
        prev = [x["debug"]["preview"] for x in r.get("photos", []) if "debug" in x]
        return _with_images(r, *prev[:2])
    except Exception as e:
        return _err(e)


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
