#!/usr/bin/env python3
"""Build per-track pick vs reject montages for harvest QA."""

from __future__ import annotations

import argparse
import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from screentime.io_utils import setup_logging


LOGGER = logging.getLogger("scripts.montage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side montage of picked vs rejected samples")
    parser.add_argument("track_dirs", nargs="+", type=Path, help="Track directories (e.g., track_0001)")
    parser.add_argument("--tiles", type=int, default=6, help="Number of tiles per row")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size in pixels")
    return parser.parse_args()


@lru_cache(maxsize=None)
def _load_sample_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def _prepare_image(path: Path, tile_size: int, header: str, footer: str) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image missing: {path}")
    resized = cv2.resize(image, (tile_size, tile_size))
    overlay = resized.copy()
    cv2.putText(overlay, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, footer, (8, tile_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _build_strip(tiles: List[np.ndarray], tile_size: int, tiles_per_row: int, label: str) -> np.ndarray:
    gap_px = 10
    label_height = 36
    blanks_needed = max(0, tiles_per_row - len(tiles))
    if blanks_needed:
        tiles.extend([np.zeros((tile_size, tile_size, 3), dtype=np.uint8) for _ in range(blanks_needed)])
    if not tiles:
        tiles = [np.zeros((tile_size, tile_size, 3), dtype=np.uint8) for _ in range(tiles_per_row)]
    gap = np.zeros((tile_size, gap_px, 3), dtype=np.uint8)
    row_img = tiles[0]
    for tile in tiles[1:]:
        row_img = np.concatenate([row_img, gap, tile], axis=1)
    label_bar = np.zeros((label_height, row_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_bar, label, (8, label_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.concatenate([label_bar, row_img], axis=0)


def _save_montage(track_dir: Path, tiles_per_row: int, tile_size: int) -> None:
    harvest_dir = track_dir.parent
    csv_path = harvest_dir / "selected_samples.csv"
    rows = _load_sample_rows(csv_path)

    try:
        track_id = int(track_dir.name.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Unable to parse track id from {track_dir}") from exc

    track_rows = [row for row in rows if int(row["track_id"]) == track_id]
    if not track_rows:
        LOGGER.warning("No rows found for %s", track_dir)
        return

    def sort_key(row: Dict[str, str]) -> float:
        try:
            return float(row.get("quality", "0"))
        except ValueError:
            return 0.0

    picked_rows = [row for row in track_rows if _parse_bool(row.get("picked", "false"))]
    picked_rows.sort(key=sort_key, reverse=True)

    rejected_rows = [
        row
        for row in track_rows
        if not _parse_bool(row.get("picked", "false")) and str(row.get("reason", "")).startswith("rejected")
    ]
    rejected_rows.sort(key=sort_key, reverse=True)
    if not rejected_rows:
        rejected_rows = [row for row in track_rows if not _parse_bool(row.get("picked", "false"))]
        rejected_rows.sort(key=sort_key, reverse=True)

    def build_tiles(source_rows: List[Dict[str, str]]) -> List[np.ndarray]:
        tiles: List[np.ndarray] = []
        for row in source_rows[:tiles_per_row]:
            path = Path(row["path"])
            reason = row.get("reason", "")
            header = reason.upper() if reason else ""
            footer = f"Q={float(row.get('quality', '0')):.2f}"
            if row.get("association_iou"):
                try:
                    footer += f" IoU={float(row['association_iou']):.2f}"
                except ValueError:
                    pass
            try:
                tiles.append(_prepare_image(path, tile_size, header, footer))
            except FileNotFoundError as err:
                LOGGER.warning(str(err))
        return tiles

    picked_tiles = build_tiles(picked_rows)
    rejected_tiles = build_tiles(rejected_rows)

    picked_strip = _build_strip(picked_tiles, tile_size, tiles_per_row, f"Track {track_id:04d} - Picked")
    rejected_strip = _build_strip(rejected_tiles, tile_size, tiles_per_row, "Rejected / Skipped")

    gap = np.zeros((20, picked_strip.shape[1], 3), dtype=np.uint8)
    montage = np.concatenate([picked_strip, gap, rejected_strip], axis=0)

    out_path = track_dir / "montage.jpg"
    cv2.imwrite(str(out_path), montage)
    LOGGER.info("Montage written to %s", out_path)


def main() -> None:
    args = parse_args()
    setup_logging()

    for track_dir in args.track_dirs:
        if not track_dir.exists():
            LOGGER.warning("Track directory missing: %s", track_dir)
            continue
        _save_montage(track_dir, args.tiles, args.tile_size)


if __name__ == "__main__":
    main()
