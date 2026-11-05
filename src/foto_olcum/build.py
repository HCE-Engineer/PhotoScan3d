"""Runs FreeCAD as a subprocess on a job file; never imports FreeCAD in this process."""
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from . import config
from .model import validate
from .project import Project

SCRIPT = Path(__file__).parent / "freecad_build" / "build_fc.py"


def run_job(job: dict, work_dir: Path, timeout: int = 300) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    job_path = work_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    result_path = Path(job["out_dir"]) / "result.json"
    result_path.unlink(missing_ok=True)
    log_path = work_dir / "freecad.log"
    env = {**os.environ, "FOTO_OLCUM_JOB": str(job_path)}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.run(config.freecad_cmd() + [str(SCRIPT)], cwd=work_dir, env=env,
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    if not result_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        return {"ok": False, "error": f"FreeCAD sonuç yazmadı (çıkış: {code})",
                "freecad_log_tail": tail, "freecad_log": str(log_path)}
    res = json.loads(result_path.read_text(encoding="utf-8"))
    res["freecad_log"] = str(log_path)
    return res


def build_project(p: Project) -> dict:
    model = p.model()
    if not model:
        return {"ok": False, "error": "model.json yok, önce set_model çağırın"}
    ms = p.measurements()
    errs = validate(model, {k: m.value_mm for k, m in ms.items()})
    if errs:
        return {"ok": False, "error": "model geçersiz", "problems": errs}
    job = {"name": p.name, "out_dir": str(p.path("out")), "model": model,
           "measurements": {k: asdict(m) for k, m in ms.items()}}
    res = run_job(job, p.path("out/_job"))
    p.log("build_freecad", {k: res.get(k) for k in ("ok", "error", "volume_mm3", "all_fully_constrained")})
    if res.get("ok"):
        p.update_state(last_build={k: res[k] for k in ("step", "stl", "fcstd", "volume_mm3", "bbox_mm")})
    return res
