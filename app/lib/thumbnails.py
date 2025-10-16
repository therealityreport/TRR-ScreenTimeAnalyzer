"""Thumbnail generation helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

from PIL import Image


def ensure_thumbnail(
    source_path: Path,
    stem: str,
    thumbnails_root: Path,
    size: Tuple[int, int] = (320, 320),
) -> Path:
    """Return a cached thumbnail path, generating it if needed."""
    thumbnails_root.mkdir(parents=True, exist_ok=True)
    dest_dir = thumbnails_root / stem
    dest_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    dest_path = dest_dir / f"{source_path.stem}_{digest}{source_path.suffix.lower()}"
    if dest_path.exists():
        src_mtime = source_path.stat().st_mtime
        dst_mtime = dest_path.stat().st_mtime
        if dst_mtime >= src_mtime:
            return dest_path
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image.thumbnail(size, Image.LANCZOS)
        image.save(dest_path, optimize=True, quality=85)
    return dest_path
