"""Project folder: photos, measurements, model, outputs and an append-only log."""
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config

SOURCES = ("photo_coarse", "photo_closeup", "caliper", "inferred", "snapped")
SUBDIRS = ("photos/raw", "debug", "masks", "out")
ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# FreeCAD rejects spreadsheet aliases that look like cell addresses or unit symbols.
CELL_RE = re.compile(r"^[A-Za-z]{1,3}[0-9]+$")
UNIT_SYMBOLS = set("""nm um mm cm dm m km mil thou in ft yd mi h min s ms us ns g mg kg t lb oz st
cwt N kN MN mN Pa kPa MPa GPa psi ksi J kJ mJ W kW mW V kV mV A mA kA K C F mF uF nF pF H
mH S T Wb Hz kHz MHz deg rad gon l ml cl L mL cal kcal eV Ws VA VAh lbf kgf M G mol cd ohm
Ohm kohm Mohm sqft cft mph kmh mph knot""".split())


@dataclass
class Measurement:
    id: str
    value_mm: float
    uncertainty_mm: float
    source: str
    label: str = ""
    locked: bool = False
    material: str = ""
    feature_ref: str = ""
    snapped_from: float | None = None
    evidence: list[str] = field(default_factory=list)
    role: str = ""          # "", "critical", "control"

    def __post_init__(self):
        if not ID_RE.match(self.id):
            raise ValueError(f"ölçü id'si geçersiz: {self.id!r} (harf/rakam/_ olmalı)")
        if self.id in UNIT_SYMBOLS or CELL_RE.match(self.id):
            raise ValueError(f"ölçü id'si {self.id!r} FreeCAD'de birim/hücre adıyla çakışıyor; "
                             "açıklayıcı bir ad verin (ör. 'plate_h', 'hole_1_dia')")
        if self.source not in SOURCES:
            raise ValueError(f"kaynak geçersiz: {self.source!r}, olası: {SOURCES}")
        if self.uncertainty_mm is None or self.uncertainty_mm < 0:
            raise ValueError(f"{self.id}: belirsizlik zorunlu ve >= 0 olmalı")


class LockedError(Exception):
    pass


class Project:
    def __init__(self, name: str):
        if not re.match(r"^[\w\-]+$", name):
            raise ValueError(f"proje adı geçersiz: {name!r} (harf, rakam, _ ve - kullanın)")
        self.name = name
        self.dir = config.projects_root() / name

    @classmethod
    def create(cls, name: str) -> "Project":
        p = cls(name)
        if p.dir.exists():
            raise FileExistsError(f"proje zaten var: {p.dir}")
        for d in SUBDIRS:
            (p.dir / d).mkdir(parents=True)
        p._write("measurements.json", {})
        p._write("photos.json", {})
        p.log("new_project", {})
        return p

    @classmethod
    def open(cls, name: str) -> "Project":
        p = cls(name)
        if not (p.dir / "measurements.json").exists():
            raise FileNotFoundError(f"proje bulunamadı: {name} ({p.dir})")
        return p

    # -- json helpers
    def _read(self, rel, default=None):
        f = self.dir / rel
        if not f.exists():
            return default
        return json.loads(f.read_text(encoding="utf-8"))

    def _write(self, rel, data):
        f = self.dir / rel
        tmp = f.with_suffix(f.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(f)

    def log(self, event: str, data: dict):
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **data},
                                ensure_ascii=False) + "\n")

    # -- measurements
    def measurements(self) -> dict[str, Measurement]:
        return {k: Measurement(**v) for k, v in self._read("measurements.json", {}).items()}

    def put_measurement(self, m: Measurement, force: bool = False):
        """Store a measurement. A locked (caliper) value is never overwritten by a
        non-caliper source; `force` only allows replacing one caliper value by another."""
        ms = self.measurements()
        old = ms.get(m.id)
        if old and old.locked and not (m.source == "caliper" and force):
            raise LockedError(f"{m.id} kumpasla kilitli ({old.value_mm} mm), değiştirilemez")
        ms[m.id] = m
        self._write("measurements.json", {k: asdict(v) for k, v in ms.items()})
        self.log("measurement", asdict(m))

    def put_photo_measurements(self, items: list[Measurement]) -> dict:
        """Bulk store from photo analysis, silently skipping locked ids."""
        ms = self.measurements()
        skipped = []
        for m in items:
            if m.id in ms and ms[m.id].locked:
                skipped.append(m.id)
                continue
            ms[m.id] = m
        self._write("measurements.json", {k: asdict(v) for k, v in ms.items()})
        self.log("photo_measurements", {"ids": [m.id for m in items], "skipped_locked": skipped})
        return {"stored": [m.id for m in items if m.id not in skipped], "skipped_locked": skipped}

    # -- photos
    def photos(self) -> dict:
        return self._read("photos.json", {})

    def add_photo(self, src: Path, kind: str) -> str:
        photos = self.photos()
        pid = f"P{len(photos) + 1:02d}"
        dst = self.dir / "photos/raw" / f"{pid}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        photos[pid] = {"file": str(dst.relative_to(self.dir)), "source": str(src), "kind": kind}
        self._write("photos.json", photos)
        return pid

    def update_photo(self, pid: str, **info):
        photos = self.photos()
        photos[pid].update(info)
        self._write("photos.json", photos)

    def path(self, rel: str) -> Path:
        return self.dir / rel

    # -- model
    def model(self):
        return self._read("model.json")

    def set_model(self, model: dict):
        self._write("model.json", model)
        self.log("set_model", {"n_features": len(model.get("features", []))})

    def state(self) -> dict:
        return self._read("state.json", {})

    def update_state(self, **kv):
        s = self.state()
        s.update(kv)
        self._write("state.json", s)
