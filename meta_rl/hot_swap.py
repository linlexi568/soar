from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
from types import SimpleNamespace

try:
    from .config import MetaRLConfig, MetaRLOutput, load_override_file
except ImportError:
    from config import MetaRLConfig, MetaRLOutput, load_override_file


def maybe_apply_overrides(
    args: Any,
    iter_idx: int,
    metrics: Optional[Dict[str, float]] = None,
    override_path: Optional[Path] = None,
) -> Any:
    """Injects overrides if a JSON file is present; otherwise returns args unchanged."""

    config = MetaRLConfig()
    path = override_path or config.override_file
    overrides = load_override_file(path)
    if overrides is None:
        return args

    # For reproducibility, tag args with meta info
    setattr(args, "meta_rl_iter", iter_idx)
    setattr(args, "meta_rl_metrics", metrics or {})
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def write_override_file(overrides: Dict[str, float], path: Optional[Path] = None) -> Path:
    config = MetaRLConfig()
    target = path or config.override_file
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    return target


def load_overrides(path: Optional[Path] = None) -> Optional[MetaRLOutput]:
    overrides = load_override_file(path or MetaRLConfig().override_file)
    if overrides is None:
        return None
    return MetaRLOutput(values=overrides)


def build_args_proxy(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)
