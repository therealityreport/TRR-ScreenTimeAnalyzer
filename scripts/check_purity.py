#!/usr/bin/env python3
"""Compute identity purity statistics for harvested tracks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency for tests
    from screentime.recognition.embed_arcface import ArcFaceEmbedder
except ImportError:  # pragma: no cover - allow importing helpers without insightface
    ArcFaceEmbedder = None  # type: ignore

LOGGER = logging.getLogger("scripts.check_purity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate harvest purity report")
    parser.add_argument("--harvest-dir", type=Path, required=True, help="Path to harvest directory")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults to <harvest-dir>/<stem>-purity.csv)",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_embedding(embedder, sample_path: Path) -> Optional[np.ndarray]:
    if not sample_path.exists():
        LOGGER.warning("Sample not found for embedding: %s", sample_path)
        return None
    import cv2  # local import to avoid unnecessary dependency when unused

    image = cv2.imread(str(sample_path))
    if image is None:
        LOGGER.warning("Failed to read sample image: %s", sample_path)
        return None
    return embedder.embed(image)


def compute_track_stats(
    entry: Dict,
    embedder,
) -> Dict[str, Optional[float]]:
    samples = entry.get("samples", [])
    if not samples:
        return {
            "sample_count": 0,
            "min_similarity": None,
            "median_similarity": None,
            "avg_similarity": None,
        }

    similarities: List[float] = []
    missing_embeddings: List[np.ndarray] = []
    for sample in samples:
        sim = sample.get("similarity_to_centroid")
        emb = None
        if sim is not None:
            similarities.append(float(sim))
        else:
            sample_path = Path(sample.get("path", ""))
            emb = ensure_embedding(embedder, sample_path)
            if emb is not None:
                missing_embeddings.append(emb)

    if missing_embeddings:
        centroid = np.mean(np.stack(missing_embeddings, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        for emb in missing_embeddings:
            similarities.append(float((emb * centroid).sum()))

    if not similarities:
        return {
            "sample_count": len(samples),
            "min_similarity": None,
            "median_similarity": None,
            "avg_similarity": None,
        }

    values = np.asarray(similarities, dtype=np.float32)
    return {
        "sample_count": len(samples),
        "min_similarity": float(values.min()),
        "median_similarity": float(np.median(values)),
        "avg_similarity": float(values.mean()),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    harvest_dir = args.harvest_dir
    manifest_path = harvest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    stem = harvest_dir.name
    report_path = args.report or (harvest_dir.parent / f"{stem}-purity.csv")

    # Determine split counts per byte track
    byte_to_tracks: Dict[Optional[int], List[int]] = {}
    for entry in manifest:
        byte_id = entry.get("byte_track_id")
        byte_to_tracks.setdefault(byte_id, []).append(entry.get("track_id"))

    if ArcFaceEmbedder is None:
        raise RuntimeError("ArcFaceEmbedder unavailable; install insightface to run purity checks")
    embedder = ArcFaceEmbedder()

    fieldnames = [
        "track_id",
        "byte_track_id",
        "sample_count",
        "min_similarity",
        "median_similarity",
        "avg_similarity",
        "byte_track_splits",
    ]

    rows: List[Dict[str, Optional[float]]] = []
    for entry in manifest:
        stats = compute_track_stats(entry, embedder)
        byte_id = entry.get("byte_track_id")
        splits = len(byte_to_tracks.get(byte_id, []))
        rows.append(
            {
                "track_id": entry.get("track_id"),
                "byte_track_id": byte_id,
                **stats,
                "byte_track_splits": splits,
            }
        )

    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Purity report written to %s", report_path)


if __name__ == "__main__":
    main()
