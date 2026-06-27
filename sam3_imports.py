"""Helpers for importing the external Meta SAM3 package.

The repository contains a documentation-only ``sam3/`` directory, so a missing
external installation can otherwise fail later with a confusing
``No module named 'sam3.model_builder'`` error.  This helper also supports the
local clone created by ``scripts/setup_sam3.sh`` (``sam3_src``) even when the
user has not run ``pip install -e sam3_src`` yet.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


def _candidate_sam3_source_paths() -> list[Path]:
    candidates = []
    env_src = os.environ.get("SAM3_SRC")
    if env_src:
        candidates.append(Path(env_src))
    candidates.extend(
        [
            PROJECT_ROOT / "sam3_src",
            PROJECT_ROOT.parent / "sam3_src",
        ]
    )
    return candidates


def _prepend_existing_sam3_sources() -> None:
    for candidate in _candidate_sam3_source_paths():
        if (candidate / "sam3" / "model_builder.py").is_file():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def import_sam3_image_components() -> Tuple[object, object]:
    """Import SAM3 image builder and processor with a helpful setup error."""
    _prepend_existing_sam3_sources()
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ModuleNotFoundError as exc:
        if exc.name and not exc.name.startswith("sam3"):
            raise
        searched = ", ".join(str(p) for p in _candidate_sam3_source_paths())
        raise ModuleNotFoundError(
            "Meta SAM3 library is not installed or is incomplete. Run "
            "`bash scripts/setup_sam3.sh` from the project root, or install the "
            "official package with `pip install -e sam3_src`. If you already "
            "cloned it elsewhere, set SAM3_SRC to that clone path. "
            f"Searched local SAM3 source paths: {searched}"
        ) from exc
    return build_sam3_image_model, Sam3Processor
