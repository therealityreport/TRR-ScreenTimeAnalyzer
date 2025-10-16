#!/usr/bin/env python3
"""Merge harvest tracks that belong to the same person using face embeddings."""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
from sklearn.cluster import DBSCAN

from screentime.io_utils import setup_logging
from screentime.recognition.embed import ArcFaceEmbedder

import logging
LOGGER = logging.getLogger("scripts.merge_tracks")


def load_track_embeddings(harvest_dir: Path, embedder: ArcFaceEmbedder):
    """Load representative embeddings for each track."""
    track_embeddings = {}
    
    for track_dir in sorted(harvest_dir.iterdir()):
        if not track_dir.is_dir() or track_dir.name == "debug":
            continue
        
        # Load a few samples from each track
        images = list(track_dir.glob("*.jpg"))[:5]  # Just use first 5
        if not images:
            continue
        
        embeddings = []
        for img_path in images:
            try:
                emb = embedder.embed_from_path(img_path)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                LOGGER.warning(f"Failed to embed {img_path}: {e}")
        
        if embeddings:
            # Use mean embedding as representative
            track_embeddings[track_dir.name] = np.mean(embeddings, axis=0)
    
    return track_embeddings


def cluster_tracks(track_embeddings: dict, similarity_threshold: float = 0.50):
    """Cluster tracks by embedding similarity."""
    track_names = list(track_embeddings.keys())
    embeddings = np.array([track_embeddings[name] for name in track_names])
    
    # Use DBSCAN with cosine similarity
    # eps = 1 - similarity_threshold (convert cosine similarity to distance)
    clustering = DBSCAN(eps=1.0 - similarity_threshold, min_samples=1, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    
    # Group tracks by cluster
    clusters = defaultdict(list)
    for track_name, label in zip(track_names, labels):
        clusters[label].append(track_name)
    
    return dict(clusters)


def merge_cluster(harvest_dir: Path, track_names: list, output_name: str):
    """Merge multiple track directories into one."""
    output_dir = harvest_dir / output_name
    output_dir.mkdir(exist_ok=True)
    
    # Copy all images from source tracks
    img_counter = 0
    for track_name in track_names:
        track_dir = harvest_dir / track_name
        for img_path in track_dir.glob("*.jpg"):
            # Rename to avoid conflicts
            new_name = f"merged_{img_counter:06d}.jpg"
            shutil.copy2(img_path, output_dir / new_name)
            img_counter += 1
    
    LOGGER.info(f"Merged {len(track_names)} tracks into {output_name} ({img_counter} images)")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge similar harvest tracks")
    parser.add_argument("harvest_dir", type=Path, help="Harvest directory")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.50,
        help="Cosine similarity threshold for merging (0-1, higher=stricter)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: harvest_dir_merged)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually merging"
    )
    args = parser.parse_args()
    
    setup_logging()
    
    if args.output_dir is None:
        args.output_dir = args.harvest_dir.parent / f"{args.harvest_dir.name}_merged"
    
    # Load embedder
    LOGGER.info("Loading face embedder...")
    embedder = ArcFaceEmbedder()
    
    # Load track embeddings
    LOGGER.info("Computing track embeddings...")
    track_embeddings = load_track_embeddings(args.harvest_dir, embedder)
    LOGGER.info(f"Loaded embeddings for {len(track_embeddings)} tracks")
    
    # Cluster tracks
    LOGGER.info(f"Clustering with similarity threshold {args.similarity_threshold}...")
    clusters = cluster_tracks(track_embeddings, args.similarity_threshold)
    
    # Report clusters
    LOGGER.info(f"\nFound {len(clusters)} unique identities:")
    for cluster_id, track_names in clusters.items():
        if len(track_names) > 1:
            LOGGER.info(f"  Cluster {cluster_id}: {track_names} -> WILL MERGE")
        else:
            LOGGER.info(f"  Cluster {cluster_id}: {track_names}")
    
    if args.dry_run:
        LOGGER.info("\n🔍 DRY RUN - No changes made")
        return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge clusters
    LOGGER.info(f"\nMerging tracks into {args.output_dir}...")
    for cluster_id, track_names in clusters.items():
        output_name = f"merged_{cluster_id:03d}"
        merge_cluster(args.harvest_dir, track_names, args.output_dir)
    
    # Copy manifest if it exists
    manifest_src = args.harvest_dir / "manifest.jsonl"
    if manifest_src.exists():
        shutil.copy2(manifest_src, args.output_dir / "manifest.jsonl")
    
    LOGGER.info(f"\n✅ Merged tracks saved to: {args.output_dir}")
    LOGGER.info(f"   Original: {len(track_embeddings)} tracks")
    LOGGER.info(f"   Merged:   {len(clusters)} identities")


if __name__ == "__main__":
    main()
