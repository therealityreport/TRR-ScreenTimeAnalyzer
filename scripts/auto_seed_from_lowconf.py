#!/usr/bin/env python3
"""Seed facebank entries automatically from low-confidence tracker hits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("scripts.auto_seed_from_lowconf")


@dataclass
class Candidate:
    label: str
    frame_idx: int
    track_id: int
    timestamp_ms: float
    lowconf_score: float
    bbox_score: float
    area_ratio: float
    laplacian_var: float
    frontal_score: float
    quality: float
    crop: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-seed facebank images from low-confidence matches.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the source video.")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Tracker annotations JSON containing frame_annotations.",
    )
    parser.add_argument(
        "--lowconf",
        type=Path,
        required=True,
        help="CSV produced by run_tracker with low-confidence per-frame records.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels to mine (e.g. LVP EILEEN).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Root facebank directory (label subdirs will be created under this).",
    )
    parser.add_argument("--min-score", type=float, default=0.55, help="Minimum low-confidence score to consider.")
    parser.add_argument("--max-per-label", type=int, default=12, help="Maximum crops to export per label.")
    parser.add_argument("--min-area", type=float, default=0.02, help="Minimum face area ratio to keep (0-1).")
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=8.0,
        help="Mean absolute pixel difference threshold for de-duplication (lower is stricter).",
    )
    return parser.parse_args(argv)


def load_annotation_index(annotations_path: Path) -> Dict[int, Dict[int, Dict]]:
    with annotations_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    index: Dict[int, Dict[int, Dict]] = {}
    for frame_entry in data.get("frame_annotations", []):
        tracks = {}
        for track in frame_entry.get("tracks", []):
            tracks[int(track["track_id"])] = track
        index[int(frame_entry["frame_idx"])] = tracks
    return index


def load_lowconf_candidates(
    lowconf_path: Path, labels: Sequence[str], min_score: float
) -> List[Tuple[int, float, int, str, float]]:
    wanted = {label.upper() for label in labels}
    candidates: List[Tuple[int, float, int, str, float]] = []
    with lowconf_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row["label"].upper()
            score = float(row["score"])
            if label not in wanted or score < min_score:
                continue
            frame_idx = int(row["frame_idx"])
            timestamp_ms = float(row.get("timestamp_ms", 0.0))
            track_id = int(row["track_id"])
            candidates.append((frame_idx, timestamp_ms, track_id, label, score))
    LOGGER.info("Loaded %d low-confidence candidates after filtering", len(candidates))
    return candidates


def compute_frontalness(landmarks: Iterable[float] | None) -> float | None:
    if landmarks is None:
        return None
    try:
        pts = np.asarray(list(landmarks), dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if pts.shape[0] < 5:
        return None
    left_eye, right_eye, nose, mouth_left, mouth_right = pts[:5]
    eye_distance = float(np.linalg.norm(left_eye - right_eye))
    if eye_distance < 1e-6:
        return None
    eye_level = abs(left_eye[1] - right_eye[1]) / eye_distance
    mouth_level = abs(mouth_left[1] - mouth_right[1]) / eye_distance
    symmetry = abs(((left_eye[0] + right_eye[0]) / 2.0) - nose[0]) / eye_distance
    frontal = 1.0 - (0.5 * symmetry + 0.3 * eye_level + 0.2 * mouth_level)
    return float(np.clip(frontal, 0.0, 1.0))


def compute_quality_metrics(
    crop: np.ndarray,
    frame_shape: Tuple[int, int],
    bbox: Sequence[float],
    bbox_score: float,
    lowconf_score: float,
    frontal_score: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    frame_h, frame_w = frame_shape
    x1, y1, x2, y2 = bbox
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    area_ratio = (width * height) / max(frame_w * frame_h, 1)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lap_norm = min(laplacian_var / 200.0, 1.0)
    area_norm = min(area_ratio / 0.25, 1.0)
    score_norm = min(max(bbox_score, lowconf_score), 1.0)
    frontal_norm = frontal_score if frontal_score is not None else 0.5

    quality = 0.55 * lap_norm + 0.2 * area_norm + 0.15 * score_norm + 0.1 * frontal_norm
    extras = {
        "laplacian_var": laplacian_var,
        "area_ratio": area_ratio,
        "score_norm": score_norm,
        "frontal_score": frontal_norm,
    }
    return quality, extras


def centre_crop_and_resize(image: np.ndarray, size: int = 128) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = image[top : top + side, left : left + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def dedup_filter(selected: List[np.ndarray], candidate: np.ndarray, threshold: float) -> bool:
    candidate_small = centre_crop_and_resize(candidate, size=64).astype(np.float32)
    for existing in selected:
        diff = np.mean(np.abs(existing - candidate_small))
        if diff < threshold:
            return False
    selected.append(candidate_small)
    return True


def export_candidates(
    grouped: Dict[str, List[Candidate]],
    out_root: Path,
    max_per_label: int,
    dedup_threshold: float,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for label, entries in grouped.items():
        label_dir = out_root / label / "AUTO"
        label_dir.mkdir(parents=True, exist_ok=True)
        entries.sort(key=lambda c: c.quality, reverse=True)
        kept: List[np.ndarray] = []
        exported = 0
        for cand in entries:
            if exported >= max_per_label:
                break
            if not dedup_filter(kept, cand.crop, dedup_threshold):
                continue
            filename = f"AUTO_f{cand.frame_idx:06d}_t{cand.track_id}_q{cand.quality:.3f}.jpg"
            path = label_dir / filename
            if not cv2.imwrite(str(path), cand.crop):
                LOGGER.warning("Failed to write %s", path)
                continue
            exported += 1
            LOGGER.info(
                "Exported %s (quality=%.3f, lap_var=%.1f, area=%.3f, frontal=%.2f, detectionscore=%.3f)",
                path,
                cand.quality,
                cand.laplacian_var,
                cand.area_ratio,
                cand.frontal_score,
                cand.bbox_score,
            )
        LOGGER.info("Label %s: exported %d/%d candidates", label, exported, len(entries))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    annotation_index = load_annotation_index(args.annotations)
    candidates = load_lowconf_candidates(args.lowconf, args.labels, args.min_score)

    if not candidates:
        LOGGER.warning("No candidates passed the filters. Nothing to do.")
        return

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    grouped: Dict[str, List[Candidate]] = defaultdict(list)
    seen_keys: set[Tuple[int, int, str]] = set()
    try:
        for frame_idx, timestamp_ms, track_id, label, score in candidates:
            key = (frame_idx, track_id, label)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            track_info = annotation_index.get(frame_idx, {}).get(track_id)
            if not track_info:
                LOGGER.debug("Missing annotation for frame=%d track=%d", frame_idx, track_id)
                continue
            bbox = track_info.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                LOGGER.warning("Failed to read frame %d from %s", frame_idx, args.video)
                continue
            frame_h, frame_w = frame.shape[:2]

            x1 = max(0, int(np.floor(bbox[0])))
            y1 = max(0, int(np.floor(bbox[1])))
            x2 = min(frame_w - 1, int(np.ceil(bbox[2])))
            y2 = min(frame_h - 1, int(np.ceil(bbox[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2].copy()

            bbox_score_val = track_info.get("score")
            bbox_score = float(bbox_score_val) if bbox_score_val is not None else float(score)
            frontal_score = compute_frontalness(track_info.get("landmarks"))
            quality, extras = compute_quality_metrics(
                crop,
                (frame_h, frame_w),
                bbox,
                bbox_score=bbox_score,
                lowconf_score=score,
                frontal_score=frontal_score,
            )
            if extras["area_ratio"] < args.min_area:
                continue

            grouped[label].append(
                Candidate(
                    label=label,
                    frame_idx=frame_idx,
                    track_id=track_id,
                    timestamp_ms=timestamp_ms,
                    lowconf_score=score,
                    bbox_score=bbox_score,
                    area_ratio=extras["area_ratio"],
                    laplacian_var=extras["laplacian_var"],
                    frontal_score=extras["frontal_score"],
                    quality=quality,
                    crop=crop,
                )
            )
    finally:
        cap.release()

    if not grouped:
        LOGGER.warning("No usable crops produced. Nothing exported.")
        return

    export_candidates(grouped, args.out_root, args.max_per_label, args.dedup_threshold)


if __name__ == "__main__":
    main()
