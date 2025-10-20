"""I/O helpers shared across CLI entrypoints and pipeline modules."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

LOGGER = logging.getLogger("screentime.io")


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file and return a dictionary."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    LOGGER.debug("Loaded YAML config %s -> keys=%s", path, list(data.keys()))
    return data


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Write YAML to disk."""
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    LOGGER.debug("Wrote YAML config %s", path)


def dump_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON to disk (with dataclass support)."""
    def _default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_default)
    LOGGER.debug("Wrote JSON file %s", path)


def load_json(path: Path) -> Any:
    """Read JSON from disk."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging if not already configured."""
    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def list_images(directory: Path) -> Iterable[Path]:
    """Yield image paths sorted in lexicographic order."""
    if not directory.exists():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )


def infer_video_stem(video_path: Path) -> str:
    """Return file stem for output naming."""
    return video_path.stem


def resolve_path(path: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p
