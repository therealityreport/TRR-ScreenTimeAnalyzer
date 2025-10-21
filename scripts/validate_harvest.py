#!/usr/bin/env python3
"""Validate harvest output and generate summary statistics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import cv2
import numpy as np

try:
    from screentime.recognition.embed_arcface import ArcFaceEmbedder as ArcFaceEmbedderImpl
except Exception:  # pragma: no cover - optional dependency for validation
    ArcFaceEmbedderImpl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from screentime.recognition.embed_arcface import ArcFaceEmbedder

LOGGER = logging.getLogger("scripts.validate_harvest")
REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate harvest output and compute metrics.")
    parser.add_argument(
        "harvest_path",
        nargs="?",
        type=Path,
        help="Path to the harvest directory (containing manifest.json or a single run subdirectory).",
    )
    parser.add_argument(
        "--harvest-dir",
        dest="harvest_dir",
        type=Path,
        default=None,
        help="Optional alias for the harvest directory path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional directory to write validation artifacts (JSON/CSV reports).",
    )
    parser.add_argument(
        "--show-tracks",
        action="store_true",
        help="Print per-track statistics to the console.",
    )
    parser.add_argument(
        "--sort-tracks-by",
        choices=("track_id", "samples", "quality"),
        default="samples",
        help="Sort key when showing per-track stats. Defaults to samples (descending).",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit with status 1 if referenced sample files are missing on disk.",
    )
    parser.add_argument(
        "--allow-extras",
        action="store_true",
        help="Permit extra image files to exist without failing validation.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Permit tracks with zero harvested samples.",
    )
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="Permit tracks containing multiple identity clusters.",
    )
    parser.add_argument(
        "--assert-max-tracks",
        type=int,
        default=24,
        help="Fail if track count exceeds this value (0 disables the check).",
    )
    args = parser.parse_args()
    harvest_dir = args.harvest_dir or args.harvest_path
    if harvest_dir is None:
        parser.error("Harvest directory required (positional argument or --harvest-dir).")
    args.harvest_dir = Path(harvest_dir).expanduser().resolve()
    return args


def ensure_manifest_dir(path: Path) -> Path:
    """Validate that the provided directory directly contains manifest.json."""
    if not path.exists():
        raise FileNotFoundError(f"Harvest directory not found: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory containing manifest.json")
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {path}")
    nested = [child for child in path.iterdir() if child.is_dir() and (child / "manifest.json").exists()]
    if nested:
        raise FileNotFoundError(
            f"Nested harvest directory detected under {path}: {[str(child) for child in nested]}"
        )
    return path


def load_manifest(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def safe_fmt(value: Optional[float], digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def resolve_sample_path(sample_path: str, harvest_dir: Path) -> Tuple[Path, bool]:
    """Resolve a sample path to an absolute path and indicate whether it exists."""
    input_path = Path(sample_path)
    candidates: List[Path] = []
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append((harvest_dir / input_path))

    for idx, part in enumerate(input_path.parts):
        if part.startswith("track_"):
            trimmed = harvest_dir / Path(*input_path.parts[idx:])
            candidates.append(trimmed)
            break

    seen: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.append(resolved)
        if resolved.exists():
            return resolved, True

    fallback = candidates[0].resolve() if candidates else input_path.resolve()
    return fallback, fallback.exists()


def collect_actual_images(harvest_dir: Path) -> Tuple[Dict[int, List[Path]], List[Path]]:
    """Gather actual image files under track directories."""
    per_track: Dict[int, List[Path]] = {}
    all_images: List[Path] = []

    for track_dir in sorted(harvest_dir.glob("track_*")):
        if not track_dir.is_dir():
            continue
        try:
            track_id = int(track_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            LOGGER.debug("Skipping non-standard track directory: %s", track_dir)
            continue

        images: List[Path] = []
        for path in track_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                resolved = path.resolve()
                images.append(resolved)
                all_images.append(resolved)
        per_track[track_id] = images

    return per_track, all_images


def cluster_embeddings(
    embeddings: Sequence[np.ndarray],
    distance_threshold: float = 0.45,
    min_cluster_size: int = 2,
) -> int:
    if not embeddings:
        return 0
    normalized = [emb / (np.linalg.norm(emb) + 1e-9) for emb in embeddings]
    clusters: List[List[np.ndarray]] = []
    centroids: List[np.ndarray] = []

    for emb in normalized:
        best_idx = None
        best_distance = float("inf")
        for idx, centroid in enumerate(centroids):
            sim = float(np.clip(np.dot(centroid, emb), -1.0, 1.0))
            distance = 1.0 - sim
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is not None and best_distance <= distance_threshold:
            clusters[best_idx].append(emb)
            centroid_vector = np.mean(clusters[best_idx], axis=0)
            centroids[best_idx] = centroid_vector / (np.linalg.norm(centroid_vector) + 1e-9)
        else:
            clusters.append([emb])
            centroids.append(emb)

    significant_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_size]
    if significant_clusters:
        return len(significant_clusters)
    return len(clusters)


def detect_multi_identity_tracks(
    manifest: Sequence[Dict],
    harvest_dir: Path,
    distance_threshold: float = 0.35,
    min_cluster_size: int = 2,
) -> List[int]:
    flagged_tracks: List[int] = []
    embedder: Optional["ArcFaceEmbedder"]
    if ArcFaceEmbedderImpl is None:
        embedder = None
    else:
        try:
            embedder = ArcFaceEmbedderImpl()
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("ArcFace embedder unavailable (%s); falling back to cosine heuristic.", exc)
            embedder = None

    for entry in manifest:
        track_id = int(entry.get("track_id"))
        samples = entry.get("samples") or []
        if len(samples) < min_cluster_size * 2:
            continue

        embeddings: List[np.ndarray] = []
        if embedder is not None:
            for sample in samples:
                path_str = sample.get("path")
                if not path_str:
                    continue
                resolved_path, exists = resolve_sample_path(path_str, harvest_dir)
                if not exists:
                    continue
                image = cv2.imread(str(resolved_path))
                if image is None:
                    LOGGER.debug("Unable to read sample image for track %s: %s", track_id, resolved_path)
                    continue
                try:
                    embedding = embedder.embed(image)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.debug("Embedding failed for %s: %s", resolved_path, exc)
                    continue
                embeddings.append(embedding)

        multi_cluster = False
        if embeddings:
            cluster_count = cluster_embeddings(embeddings, distance_threshold, min_cluster_size)
            multi_cluster = cluster_count > 1
        else:
            cos_values = [sample.get("identity_cosine") for sample in samples if sample.get("identity_cosine") is not None]
            if cos_values:
                cos_threshold = 1.0 - distance_threshold
                high = sum(1 for value in cos_values if float(value) >= cos_threshold)
                low = len(cos_values) - high
                multi_cluster = high >= min_cluster_size and low >= min_cluster_size

        if multi_cluster:
            flagged_tracks.append(track_id)

    return flagged_tracks


@dataclass
class TrackSummary:
    track_id: int
    byte_track_id: Optional[int]
    label: Optional[str]
    sample_count: int
    missing_samples: int
    directory_images: int
    duration_seconds: Optional[float]
    total_frames: Optional[int]
    avg_quality: Optional[float]
    median_quality: Optional[float]
    avg_sharpness: Optional[float]
    avg_frontalness: Optional[float]
    avg_score: Optional[float]


def summarise_tracks(
    manifest: Sequence[Dict],
    harvest_dir: Path,
) -> Tuple[
    List[TrackSummary],
    List[Tuple[int, str, Path]],
    List[Path],
    Dict[str, float],
]:
    """Generate per-track summaries and aggregate metrics."""
    actual_images_by_track, actual_images = collect_actual_images(harvest_dir)

    all_quality: List[float] = []
    all_sharpness: List[float] = []
    all_frontalness: List[float] = []
    all_scores: List[float] = []
    samples_per_track: List[int] = []

    referenced_paths: List[Path] = []
    existing_paths: List[Path] = []
    missing_samples: List[Tuple[int, str, Path]] = []

    summaries: List[TrackSummary] = []
    track_ids_in_manifest = set()

    for entry in manifest:
        track_id = int(entry.get("track_id"))
        track_ids_in_manifest.add(track_id)
        samples = entry.get("samples") or []
        sample_count = len(samples)
        samples_per_track.append(sample_count)

        quality_values: List[float] = []
        sharpness_values: List[float] = []
        frontalness_values: List[float] = []
        score_values: List[float] = []
        missing_for_track = 0

        for sample in samples:
            path_str = sample.get("path")
            if not path_str:
                missing_for_track += 1
                continue

            resolved_path, exists = resolve_sample_path(path_str, harvest_dir)
            referenced_paths.append(resolved_path)
            if exists:
                existing_paths.append(resolved_path)
            else:
                missing_samples.append((track_id, path_str, resolved_path))
                missing_for_track += 1

            quality = sample.get("quality")
            if quality is not None:
                quality_values.append(float(quality))
                all_quality.append(float(quality))

            sharpness = sample.get("sharpness")
            if sharpness is not None:
                sharpness_values.append(float(sharpness))
                all_sharpness.append(float(sharpness))

            frontalness = sample.get("frontalness")
            if frontalness is not None:
                frontalness_values.append(float(frontalness))
                all_frontalness.append(float(frontalness))

            score = sample.get("score")
            if score is not None:
                score_values.append(float(score))
                all_scores.append(float(score))

        # Track-level aggregates
        duration_seconds = None
        first_ts = entry.get("first_ts_ms")
        last_ts = entry.get("last_ts_ms")
        if first_ts is not None and last_ts is not None:
            duration_seconds = float(last_ts - first_ts) / 1000.0

        avg_quality = safe_mean(quality_values)
        median_quality = safe_median(quality_values)
        avg_sharpness = safe_mean(sharpness_values)
        avg_frontalness = safe_mean(frontalness_values)
        avg_score = safe_mean(score_values)

        directory_images = len(actual_images_by_track.get(track_id, []))

        summaries.append(
            TrackSummary(
                track_id=track_id,
                byte_track_id=entry.get("byte_track_id"),
                label=entry.get("label"),
                sample_count=sample_count,
                missing_samples=missing_for_track,
                directory_images=directory_images,
                duration_seconds=duration_seconds,
                total_frames=entry.get("total_frames"),
                avg_quality=avg_quality,
                median_quality=median_quality,
                avg_sharpness=avg_sharpness,
                avg_frontalness=avg_frontalness,
                avg_score=avg_score,
            )
        )

    aggregate_metrics = {
        "total_tracks": len(manifest),
        "total_samples": sum(samples_per_track),
        "samples_per_track_mean": safe_mean(samples_per_track),
        "samples_per_track_median": safe_median(samples_per_track),
        "quality_mean": safe_mean(all_quality),
        "quality_median": safe_median(all_quality),
        "sharpness_mean": safe_mean(all_sharpness),
        "frontalness_mean": safe_mean(all_frontalness),
        "score_mean": safe_mean(all_scores),
        "referenced_files": len(referenced_paths),
        "existing_files": len(existing_paths),
    }

    extra_images = [path for path in actual_images if path not in existing_paths]

    # Identify track directories missing from manifest for additional reporting
    extra_track_dirs = sorted(set(actual_images_by_track) - track_ids_in_manifest)
    missing_track_dirs = sorted(track_ids_in_manifest - set(actual_images_by_track))
    aggregate_metrics["extra_track_dirs"] = extra_track_dirs
    aggregate_metrics["missing_track_dirs"] = missing_track_dirs

    return summaries, missing_samples, extra_images, aggregate_metrics


def write_reports(
    output_dir: Path,
    summaries: Sequence[TrackSummary],
    missing_samples: Sequence[Tuple[int, str, Path]],
    extra_images: Sequence[Path],
    aggregate_metrics: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        **aggregate_metrics,
        "missing_samples": [
            {"track_id": track_id, "manifest_path": manifest_path, "resolved_path": str(resolved)}
            for track_id, manifest_path, resolved in missing_samples
        ],
        "extra_images": [str(path) for path in extra_images],
        "multi_identity_tracks": [int(tid) for tid in aggregate_metrics.get("multi_identity_tracks", [])],
        "tracks_with_zero_picks": [int(tid) for tid in aggregate_metrics.get("tracks_with_zero_picks", [])],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    with (output_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "track_id",
            "byte_track_id",
            "label",
            "sample_count",
            "missing_samples",
            "directory_images",
            "duration_seconds",
            "total_frames",
            "avg_quality",
            "median_quality",
            "avg_sharpness",
            "avg_frontalness",
            "avg_score",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)

    if missing_samples:
        with (output_dir / "missing_samples.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["track_id", "manifest_path", "resolved_path"], dialect="excel"
            )
            writer.writeheader()
            for track_id, manifest_path, resolved in missing_samples:
                writer.writerow(
                    {
                        "track_id": track_id,
                        "manifest_path": manifest_path,
                        "resolved_path": str(resolved),
                    }
                )

    if extra_images:
        with (output_dir / "extra_images.txt").open("w", encoding="utf-8") as fh:
            for path in extra_images:
                fh.write(f"{path}\n")


def print_summary(
    harvest_dir: Path,
    summaries: Sequence[TrackSummary],
    missing_samples: Sequence[Tuple[int, str, Path]],
    extra_images: Sequence[Path],
    aggregate_metrics: Dict[str, float],
) -> None:
    total_tracks = aggregate_metrics["total_tracks"]
    labeled_tracks = sum(1 for summary in summaries if summary.label)
    empty_tracks = sum(1 for summary in summaries if summary.sample_count == 0)
    total_samples = aggregate_metrics["total_samples"]
    avg_samples = aggregate_metrics["samples_per_track_mean"]
    median_samples = aggregate_metrics["samples_per_track_median"]

    existing_files = aggregate_metrics["existing_files"]
    referenced_files = aggregate_metrics["referenced_files"]

    quality_mean = aggregate_metrics["quality_mean"]
    sharpness_mean = aggregate_metrics["sharpness_mean"]
    frontalness_mean = aggregate_metrics["frontalness_mean"]

    extra_track_dirs = aggregate_metrics.get("extra_track_dirs", [])
    missing_track_dirs = aggregate_metrics.get("missing_track_dirs", [])
    multi_identity_tracks = aggregate_metrics.get("multi_identity_tracks", [])

    print(f"Validated harvest directory: {harvest_dir}")
    print("")
    print("STATISTICS:")
    print(f"Total tracks:          {total_tracks}")
    print(f"  Labeled:             {labeled_tracks}")
    print(f"  Unlabeled:           {total_tracks - labeled_tracks}")
    print(f"  Empty tracks:        {empty_tracks}")
    if multi_identity_tracks:
        sorted_ids = sorted(int(tid) for tid in multi_identity_tracks)
        print(f"  Multi-identity:      {len(sorted_ids)} -> {sorted_ids}")
    if missing_track_dirs:
        print(f"  Missing track dirs:  {len(missing_track_dirs)} -> {missing_track_dirs}")
    if extra_track_dirs:
        print(f"  Extra track dirs:    {len(extra_track_dirs)} -> {extra_track_dirs}")

    print("")
    print("SAMPLES:")
    print(f"Total samples (manifest): {total_samples}")
    print(f"Unique referenced files:   {referenced_files}")
    print(f"Existing referenced files: {existing_files}")
    print(f"Missing referenced files:  {len(missing_samples)}")
    print(f"Actual image files:        {existing_files + len(extra_images)}")
    print(f"Extra image files:         {len(extra_images)}")
    print(f"Average samples/track:     {safe_fmt(avg_samples, 2)}")
    print(f"Median samples/track:      {safe_fmt(median_samples, 2)}")

    print("")
    print("QUALITY METRICS:")
    print(f"  Average quality:     {safe_fmt(quality_mean)}")
    print(f"  Average sharpness:   {safe_fmt(sharpness_mean)}")
    print(f"  Average frontalness: {safe_fmt(frontalness_mean)}")

    if missing_samples:
        print("")
        print("WARNINGS:")
        print("  Missing sample files detected; inspect missing_samples.csv or manifest paths.")


def print_track_table(summaries: Sequence[TrackSummary], sort_key: str) -> None:
    if not summaries:
        print("No track summaries to display.")
        return

    reverse = sort_key in {"samples", "quality"}
    if sort_key == "samples":
        key_fn = lambda item: item.sample_count
    elif sort_key == "quality":
        key_fn = lambda item: item.avg_quality or -1.0
    else:
        key_fn = lambda item: item.track_id

    rows = sorted(summaries, key=key_fn, reverse=reverse)
    header = (
        "Track  Samples  Missing  DirFiles  AvgQ   AvgSharp  AvgFront  Duration(s)  Label"
    )
    print("")
    print("TRACK SUMMARY:")
    print(header)
    for summary in rows:
        print(
            f"{summary.track_id:5d}  "
            f"{summary.sample_count:7d}  "
            f"{summary.missing_samples:7d}  "
            f"{summary.directory_images:8d}  "
            f"{safe_fmt(summary.avg_quality, 2):>5}  "
            f"{safe_fmt(summary.avg_sharpness, 1):>8}  "
            f"{safe_fmt(summary.avg_frontalness, 2):>8}  "
            f"{safe_fmt(summary.duration_seconds, 1):>11}  "
            f"{summary.label or '-'}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    harvest_dir = ensure_manifest_dir(args.harvest_dir)
    manifest_path = harvest_dir / "manifest.json"
    manifest = load_manifest(manifest_path)

    summaries, missing_samples, extra_images, aggregate_metrics = summarise_tracks(manifest, harvest_dir)

    multi_identity_tracks = detect_multi_identity_tracks(manifest, harvest_dir)
    aggregate_metrics["multi_identity_tracks"] = multi_identity_tracks
    empty_track_ids = [summary.track_id for summary in summaries if summary.sample_count == 0]
    aggregate_metrics["tracks_with_zero_picks"] = empty_track_ids

    print_summary(harvest_dir, summaries, missing_samples, extra_images, aggregate_metrics)

    if args.show_tracks:
        print_track_table(summaries, args.sort_tracks_by)

    if args.output:
        write_reports(args.output, summaries, missing_samples, extra_images, aggregate_metrics)
        print(f"\nReports written to {args.output}")

    exit_code = 0

    if args.fail_on_missing and missing_samples:
        LOGGER.error("Missing referenced sample files detected; rerun without --fail-on-missing to ignore.")
        exit_code = 1

    if extra_images and not args.allow_extras:
        LOGGER.error("Extra image files detected; rerun with --allow-extras to ignore.")
        exit_code = 1

    if empty_track_ids and not args.allow_empty:
        LOGGER.error("Tracks with zero samples detected: %s", empty_track_ids)
        exit_code = 1

    if multi_identity_tracks and not args.allow_mixed:
        LOGGER.error("Multi-identity tracks detected: %s", multi_identity_tracks)
        exit_code = 1

    if args.assert_max_tracks and args.assert_max_tracks > 0:
        total_tracks = aggregate_metrics.get("total_tracks", 0)
        if total_tracks > args.assert_max_tracks:
            LOGGER.error(
                "Track count %s exceeds --assert-max-tracks=%s",
                total_tracks,
                args.assert_max_tracks,
            )
            exit_code = 1

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
