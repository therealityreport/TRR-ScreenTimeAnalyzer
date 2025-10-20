#!/usr/bin/env python3
"""Cluster harvested tracks by identity using ArcFace embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.types import l2_normalize


LOGGER = logging.getLogger("scripts.cluster_tracks")


@dataclass
class TrackSamples:
    track_id: int
    paths: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster harvested tracks using ArcFace embeddings.")
    parser.add_argument("harvest_dir", type=Path, help="Harvest directory containing manifest.json and crops.")
    parser.add_argument(
        "--cluster-thresh",
        type=float,
        default=0.45,
        help="Cosine distance threshold for agglomerative clustering (default 0.45).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum picked samples required for a track to be clustered (default 1).",
    )
    parser.add_argument(
        "--arcface-model",
        type=str,
        default=None,
        help="Optional ArcFace model path/name for embeddings.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=None,
        help="Optional ONNX providers (e.g. CoreMLExecutionProvider CPUExecutionProvider).",
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=None,
        help="Optional override for selected_samples.csv path.",
    )
    return parser.parse_args()


def load_selected_samples(harvest_dir: Path, csv_path: Path, min_samples: int) -> List[TrackSamples]:
    df = pd.read_csv(csv_path)
    if "picked" in df.columns:
        df = df[df["picked"].astype(str).str.lower().isin({"1", "true", "yes"})]
    if "is_debug" in df.columns:
        df = df[df["is_debug"].astype(str).str.lower().isin({"0", "false", "nan"})]
    required_cols = {"track_id", "path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"{csv_path} missing required columns: {sorted(missing_cols)}")

    tracks: List[TrackSamples] = []
    for track_id, group in df.groupby("track_id"):
        paths: List[Path] = []
        for raw_path in group["path"]:
            rel_path = Path(str(raw_path))
            path = rel_path if rel_path.is_absolute() else harvest_dir / rel_path
            if not path.exists():
                LOGGER.warning("Skipping missing sample %s for track %s", path, track_id)
                continue
            paths.append(path)
        if len(paths) < max(1, min_samples):
            LOGGER.debug("Track %s has insufficient picked samples (%d); skipping.", track_id, len(paths))
            continue
        tracks.append(TrackSamples(track_id=int(track_id), paths=paths))
    tracks.sort(key=lambda item: item.track_id)
    return tracks


def embed_samples(tracks: Sequence[TrackSamples], embedder: ArcFaceEmbedder) -> Dict[int, np.ndarray]:
    track_embeddings: Dict[int, np.ndarray] = {}
    for track in tracks:
        embeddings: List[np.ndarray] = []
        for path in track.paths:
            image = cv2.imread(str(path))
            if image is None:
                LOGGER.warning("Failed to read %s for track %s; skipping sample.", path, track.track_id)
                continue
            try:
                embedding = embedder.embed(image)
            except Exception as exc:  # pragma: no cover - runtime guard
                LOGGER.warning("Embedding failed for %s: %s", path, exc)
                continue
            embeddings.append(embedding)
        if not embeddings:
            LOGGER.warning("Track %s produced no embeddings; skipping.", track.track_id)
            continue
        stacked = np.stack(embeddings, axis=0)
        if stacked.shape[0] == 1:
            medoid = stacked[0]
        else:
            # cosine distance = 1 - cosine similarity
            sims = stacked @ stacked.T
            sims = np.clip(sims, -1.0, 1.0)
            dists = 1.0 - sims
            medoid_index = int(np.argmin(dists.sum(axis=1)))
            medoid = stacked[medoid_index]
        track_embeddings[track.track_id] = l2_normalize(medoid.astype(np.float32))
    return track_embeddings


def cluster_tracks(embeddings: Dict[int, np.ndarray], distance_thresh: float) -> Dict[int, List[int]]:
    if not embeddings:
        return {}
    track_ids = sorted(embeddings.keys())
    matrix = np.stack([embeddings[tid] for tid in track_ids], axis=0)
    if len(track_ids) == 1:
        return {0: [track_ids[0]]}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=max(1e-6, float(distance_thresh)),
    )
    labels = clustering.fit_predict(matrix)
    label_map: Dict[int, List[int]] = {}
    for track_id, label in zip(track_ids, labels):
        label_map.setdefault(int(label), []).append(track_id)
    # Reindex labels to sequential ids starting at 0
    clusters: Dict[int, List[int]] = {}
    for new_id, label in enumerate(sorted(label_map.keys())):
        clusters[new_id] = sorted(label_map[label])
    return clusters


def write_clusters(harvest_dir: Path, clusters: Dict[int, List[int]]) -> Path:
    payload = {
        "clusters": [
            {"id": cluster_id, "tracks": tracks}
            for cluster_id, tracks in sorted(clusters.items(), key=lambda item: item[0])
        ]
    }
    output_path = harvest_dir / "clusters.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    harvest_dir = args.harvest_dir.expanduser().resolve()
    if not harvest_dir.exists():
        raise SystemExit(f"Harvest directory not found: {harvest_dir}")
    manifest_path = harvest_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found in {harvest_dir}")

    csv_path = args.selected_csv or (harvest_dir / "selected_samples.csv")
    if not csv_path.exists():
        raise SystemExit(f"selected_samples.csv not found in {csv_path.parent}")

    sample_tracks = load_selected_samples(harvest_dir, csv_path, args.min_samples)
    if not sample_tracks:
        raise SystemExit("No tracks with picked samples available for clustering.")

    LOGGER.info("Loaded %d tracks with picked samples.", len(sample_tracks))
    embedder = ArcFaceEmbedder(model_path=args.arcface_model, providers=args.providers)
    track_embeddings = embed_samples(sample_tracks, embedder)
    if not track_embeddings:
        raise SystemExit("Failed to compute embeddings for any track.")

    clusters = cluster_tracks(track_embeddings, args.cluster_thresh)
    output_path = write_clusters(harvest_dir, clusters)
    LOGGER.info("Wrote %d clusters to %s", len(clusters), output_path)


if __name__ == "__main__":
    main()
