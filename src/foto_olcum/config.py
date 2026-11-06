import os
from pathlib import Path

DEFAULT_FREECAD = Path.home() / "Applications" / "FreeCAD_1.1.4.AppImage"


def projects_root() -> Path:
    root = Path(os.environ.get("FOTO_OLCUM_PROJECTS", Path.home() / "foto-olcum-projeler"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def calibration_dir() -> Path:
    d = projects_root() / "_kalibrasyon"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sheets_dir() -> Path:
    d = projects_root() / "_altliklar"
    d.mkdir(parents=True, exist_ok=True)
    return d


def freecad_cmd() -> list[str]:
    """Command prefix that runs a FreeCAD script headless.

    FOTO_OLCUM_FREECAD may point at an AppImage (run as `<appimage> freecadcmd`) or at a
    freecadcmd binary."""
    path = Path(os.environ.get("FOTO_OLCUM_FREECAD", DEFAULT_FREECAD)).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"FreeCAD bulunamadı: {path}. FOTO_OLCUM_FREECAD ortam değişkenini ayarlayın.")
    if path.suffix.lower() == ".appimage":
        return [str(path), "freecadcmd"]
    return [str(path)]
