from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def make_run_dir(project_name: str, outputs_root: str | Path = "outputs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = _git_short_sha()
    run_dir = Path(outputs_root) / project_name / f"{ts}-{sha}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
