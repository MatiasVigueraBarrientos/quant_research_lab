from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping (dict) at top level.")
    return data
