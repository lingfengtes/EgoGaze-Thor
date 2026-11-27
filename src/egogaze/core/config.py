"""YAML configuration loader with sensible defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path:
        cfg_path = Path(path)
    else:
        local_cfg = project_root() / "configs" / "local.yaml"
        cfg_path = local_cfg if local_cfg.exists() else project_root() / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
