#!/usr/bin/env python3
"""Cluster harvested tracks by identity using ArcFace embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
        "--sim",
        dest="cluster_thresh",
        type=float,
        default=0.45,
        help="Cosine distance threshold for agglomerative clustering (alias: --sim).",
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
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=None,
        help="Minimum sharpness required for picked samples (filters selected_samples.csv).",
    )
    parser.add_argument(
        "--min-area-frac",
        type=float,
        default=None,
        help="Minimum face area fraction required for picked samples.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Minimum number of tracks required to keep a cluster in the output (default 1).",
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=None,
        help="Optional path for a JSON QC report (defaults to harvest_dir/clusters_qc.json).",
    )
    return parser.parse_args()


def load_selected_samples(
    harvest_dir: Path,
    csv_path: Path,
    min_samples: int,
    min_sharpness: Optional[float] = None,
    min_area_frac: Optional[float] = None,
) -> List[TrackSamples]:
    df = pd.read_csv(csv_path)
    if "picked" in df.columns:
        df = df[df["picked"].astype(str).str.lower().isin({"1", "true", "yes"})]
    if "is_debug" in df.columns:
        df = df[df["is_debug"].astype(str).str.lower().isin({"0", "false", "nan"})]

    if min_sharpness is not None and "sharpness" in df.columns:
        before = len(df)
        df = df[df["sharpness"].astype(float) >= float(min_sharpness)]
        dropped = before - len(df)
        if dropped:
            LOGGER.info("Filtered %d samples below sharpness %.2f", dropped, min_sharpness)

    if min_area_frac is not None and "area_frac" in df.columns:
        before = len(df)
        df = df[df["area_frac"].astype(float) >= float(min_area_frac)]
        dropped = before - len(df)
        if dropped:
            LOGGER.info("Filtered %d samples below area fraction %.3f", dropped, min_area_frac)

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


def _cluster_metrics(cluster_id: int, track_ids: List[int], embeddings: Dict[int, np.ndarray]) -> Dict[str, object]:
    vectors = [embeddings[tid] for tid in track_ids if tid in embeddings]
    metrics: Dict[str, object] = {
        "original_cluster_id": cluster_id,
        "size": len(track_ids),
        "tracks": track_ids,
    }
    if len(vectors) < 2:
        metrics.update(
            {
                "pairwise_count": 0,
                "mean_similarity": 1.0,
                "min_similarity": 1.0,
                "max_similarity": 1.0,
                "variance": 0.0,
            }
        )
        return metrics

    stacked = np.stack(vectors, axis=0)
    sims = np.clip(stacked @ stacked.T, -1.0, 1.0)
    idx = np.triu_indices(sims.shape[0], k=1)
    pairwise = sims[idx]
    metrics.update(
        {
            "pairwise_count": int(pairwise.size),
            "mean_similarity": float(np.mean(pairwise)),
            "min_similarity": float(np.min(pairwise)),
            "max_similarity": float(np.max(pairwise)),
            "variance": float(np.var(pairwise)),
        }
    )
    return metrics


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

    sample_tracks = load_selected_samples(
        harvest_dir,
        csv_path,
        args.min_samples,
        min_sharpness=args.min_sharpness,
        min_area_frac=args.min_area_frac,
    )
    if not sample_tracks:
        raise SystemExit("No tracks with picked samples available for clustering.")

    LOGGER.info("Loaded %d tracks with picked samples.", len(sample_tracks))
    embedder = ArcFaceEmbedder(model_path=args.arcface_model, providers=args.providers)
    track_embeddings = embed_samples(sample_tracks, embedder)
    if not track_embeddings:
        raise SystemExit("Failed to compute embeddings for any track.")

    clusters = cluster_tracks(track_embeddings, args.cluster_thresh)
    kept_clusters: Dict[int, List[int]] = {}
    qc_clusters: List[Dict[str, object]] = []
    flagged_clusters: List[Dict[str, object]] = []
    next_cluster_id = 0
    for original_id, tracks in sorted(clusters.items(), key=lambda item: item[0]):
        metrics = _cluster_metrics(original_id, tracks, track_embeddings)
        if len(tracks) >= max(1, args.min_cluster_size):
            metrics["cluster_id"] = next_cluster_id
            kept_clusters[next_cluster_id] = tracks
            qc_clusters.append(metrics)
            next_cluster_id += 1
        else:
            metrics["cluster_id"] = None
            flagged_clusters.append(metrics)

    if flagged_clusters:
        LOGGER.info(
            "Flagged %d clusters smaller than min size %d.",
            len(flagged_clusters),
            args.min_cluster_size,
        )

    output_path = write_clusters(harvest_dir, kept_clusters)
    LOGGER.info("Wrote %d clusters to %s", len(kept_clusters), output_path)

    qc_path = args.qc_report or (harvest_dir / "clusters_qc.json")
    qc_payload = {
        "parameters": {
            "cluster_threshold": args.cluster_thresh,
            "min_cluster_size": args.min_cluster_size,
            "min_samples_per_track": args.min_samples,
            "min_sharpness": args.min_sharpness,
            "min_area_frac": args.min_area_frac,
        },
        "clusters": qc_clusters,
        "flagged_clusters": flagged_clusters,
    }
    with qc_path.open("w", encoding="utf-8") as fh:
        json.dump(qc_payload, fh, indent=2)
    LOGGER.info("Cluster QC report written to %s", qc_path)


if __name__ == "__main__":
    main()
