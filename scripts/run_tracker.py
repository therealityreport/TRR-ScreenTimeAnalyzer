#!/usr/bin/env python3
"""CLI for running YOLO + ByteTrack + ArcFace recognition pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import pandas as pd

from screentime.attribution import aggregate, associate
from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.io_utils import dump_json, ensure_dir, infer_video_stem, load_yaml, setup_logging
from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.recognition.facebank import load_facebank
from screentime.recognition.matcher import TrackVotingMatcher
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper, TrackAccumulator
from screentime.types import TrackState, bbox_area, iou


LOGGER = logging.getLogger("scripts.run_tracker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full tracking and recognition pipeline")
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/outputs"),
        help="Output directory root",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Pipeline configuration YAML",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path("configs/bytetrack.yaml"),
        help="ByteTrack configuration YAML",
    )
    parser.add_argument(
        "--person-weights",
        type=str,
        required=True,
        help="YOLO person detector weights",
    )
    parser.add_argument(
        "--facebank-parquet",
        type=Path,
        default=Path("data/facebank.parquet"),
        help="Path to facebank parquet file",
    )
    parser.add_argument(
        "--arcface-model",
        type=str,
        default=None,
        help="Optional ArcFace model override",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=None,
        help="ONNX execution providers (overrides platform defaults)",
    )
    parser.add_argument(
        "--retina-det-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Override RetinaFace detection size",
    )
    parser.add_argument("--face-det-threshold", type=float, default=None)
    parser.add_argument("--person-conf", type=float, default=None)
    identity_group = parser.add_mutually_exclusive_group()
    identity_group.add_argument(
        "--identity-split-enabled",
        dest="identity_split",
        action="store_true",
        help="Force enable identity splitting within tracks",
    )
    identity_group.add_argument(
        "--no-identity-split",
        dest="identity_split",
        action="store_false",
        help="Disable identity splitting within tracks",
    )
    identity_group.set_defaults(identity_split=None)
    parser.add_argument(
        "--identity-split-min-frames",
        type=int,
        default=None,
        help="Frames required to confirm label change",
    )
    parser.add_argument(
        "--identity-change-margin",
        type=float,
        default=None,
        help="Cosine margin for identity change",
    )
    parser.add_argument(
        "--max-gap-ms",
        type=float,
        default=None,
        help="Max gap to bridge within segments (ms)",
    )
    parser.add_argument(
        "--min-run-ms",
        type=float,
        default=None,
        help="Minimum segment duration (ms)",
    )
    parser.add_argument(
        "--similarity-th",
        type=float,
        default=None,
        help="Override matcher similarity threshold",
    )
    parser.add_argument(
        "--embed-every-n",
        type=int,
        default=None,
        help="Compute ArcFace embeddings every N processed frames (default from config)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override frame sampling stride",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    _configure_runtime_threads()

    pipeline_cfg = load_yaml(args.pipeline_config)
    tracker_cfg = load_yaml(args.tracker_config)

    stride = args.stride if args.stride is not None else int(pipeline_cfg.get("stride", 1))
    if stride < 1:
        LOGGER.warning("Invalid stride %s requested; defaulting to 1", stride)
        stride = 1

    det_size = tuple(args.retina_det_size) if args.retina_det_size else tuple(pipeline_cfg.get("det_size", [960, 960]))
    face_conf = args.face_det_threshold if args.face_det_threshold is not None else pipeline_cfg.get("face_conf_th", 0.45)
    person_conf = args.person_conf if args.person_conf is not None else pipeline_cfg.get("person_conf_th", 0.20)
    providers = _resolve_providers(args.providers, pipeline_cfg.get("providers"))

    person_detector = YOLOPersonDetector(weights=args.person_weights, conf_thres=person_conf)
    face_detector = RetinaFaceDetector(providers=providers, det_size=det_size, det_thresh=face_conf)
    tracker = ByteTrackWrapper(**tracker_cfg)

    embedder = ArcFaceEmbedder(model_path=args.arcface_model, providers=providers)
    facebank_embeddings = load_facebank(args.facebank_parquet)
    
    # Log facebank information
    facebank_labels = sorted(facebank_embeddings.keys())
    LOGGER.info(
        "Loaded facebank: %d identities → %s",
        len(facebank_labels),
        facebank_labels,
    )
    
    similarity_th = args.similarity_th if args.similarity_th is not None else pipeline_cfg.get("similarity_th", 0.82)
    embed_every_n_setting = args.embed_every_n if args.embed_every_n is not None else pipeline_cfg.get("embed_every_n", 2)
    embed_every_n = int(embed_every_n_setting or 1)
    if embed_every_n < 1:
        LOGGER.warning("Invalid embed_every_n=%s requested; defaulting to 1", embed_every_n_setting)
        embed_every_n = 1
    identity_split_enabled = (
        args.identity_split if args.identity_split is not None else pipeline_cfg.get("identity_split_enabled", True)
    )
    # CLI flag overrides config; if not set on CLI, use config default (true)
    identity_split_min_frames = args.identity_split_min_frames if args.identity_split_min_frames is not None else pipeline_cfg.get("identity_split_min_frames", 5)
    identity_change_margin = args.identity_change_margin if args.identity_change_margin is not None else pipeline_cfg.get("identity_change_margin", 0.05)
    vote_decay = pipeline_cfg.get("vote_decay", 0.99)
    flip_tolerance = pipeline_cfg.get("flip_tolerance", 0.30)
    dilate_track_px = float(pipeline_cfg.get("dilate_track_px", 0.0))
    face_in_track_iou = float(pipeline_cfg.get("face_in_track_iou", 0.30))

    LOGGER.info(
        "Runtime config: providers=%s stride=%d embed_every_n=%d person_conf=%.2f face_conf=%.2f",
        providers,
        stride,
        embed_every_n,
        person_conf,
        face_conf,
    )

    LOGGER.info(
        "Identity-split config: enabled=%s min_frames=%d change_margin=%.3f",
        identity_split_enabled,
        identity_split_min_frames,
        identity_change_margin,
    )
    LOGGER.info(
        "Matching config: similarity_th=%.3f vote_decay=%.3f flip_tolerance=%.3f",
        similarity_th,
        vote_decay,
        flip_tolerance,
    )
    LOGGER.info(
        "Association config: face_in_track_iou=%.2f dilate_track_px=%.3f",
        face_in_track_iou,
        dilate_track_px,
    )
    
    matcher = TrackVotingMatcher(
        facebank_embeddings,
        similarity_th=similarity_th,
        vote_decay=vote_decay,
        flip_tolerance=flip_tolerance,
        identity_split_enabled=identity_split_enabled,
        identity_split_min_frames=identity_split_min_frames,
        identity_change_margin=identity_change_margin,
    )

    video_path = args.video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker.set_frame_rate(fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = float(width * height)

    LOGGER.info(
        "Running tracker video=%s fps=%.2f frames=%s facebank=%s",
        video_path,
        fps,
        frame_count or "unknown",
        args.facebank_parquet,
    )

    accumulator = TrackAccumulator()
    track_similarity: Dict[int, float] = {}
    frame_annotations: Dict[int, List[Dict]] = {}

    frame_idx = -1
    processed_frames = 0
    if frame_count:
        progress_step = max(1, frame_count // 20)  # ~5% increments
    else:
        progress_step = 500
    next_progress_log = progress_step
    last_frames_seen = 0

    def log_progress(frames_seen: int) -> None:
        nonlocal next_progress_log
        if frames_seen < next_progress_log:
            return
        pct = (frames_seen / frame_count * 100.0) if frame_count else None
        active_tracks = len(accumulator.active)
        finished_tracks = len(accumulator.finished)
        labelled_tracks = sum(1 for track in accumulator.active.values() if track.label) + sum(
            1 for track in accumulator.finished if track.label
        )
        if pct is not None:
            LOGGER.info(
                "Tracker progress %.1f%% (%d/%d frames seen, processed=%d, active=%d, finished=%d, labelled=%d)",
                pct,
                frames_seen,
                frame_count,
                processed_frames,
                active_tracks,
                finished_tracks,
                labelled_tracks,
            )
        else:
            LOGGER.info(
                "Tracker progress: %d frames seen (processed=%d, active=%d, finished=%d, labelled=%d)",
                frames_seen,
                processed_frames,
                active_tracks,
                finished_tracks,
                labelled_tracks,
            )
        while next_progress_log <= frames_seen:
            next_progress_log += progress_step

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frames_seen = frame_idx + 1
        if frame_idx % stride != 0:
            log_progress(frames_seen)
            continue

        processed_frames += 1
        timestamp_ms = (frame_idx / fps) * 1000.0
        person_dets = person_detector.detect(frame, frame_idx)
        observations = tracker.update(person_dets, frame.shape)
        accumulator.update(frame_idx, timestamp_ms, observations)

        track_lookup = {obs.track_id: obs for obs in observations}
        face_dets = face_detector.detect(frame, frame_idx)

        frame_boxes: List[Dict] = []
        for obs in observations:
            state = accumulator.active.get(obs.track_id)
            label = state.label if state else None
            score = track_similarity.get(obs.track_id)
            frame_boxes.append(
                {
                    "track_id": obs.track_id,
                    "bbox": list(map(float, obs.bbox)),
                    "label": label,
                    "score": score,
                }
            )

        for face in face_dets:
            assigned_track = _associate_face(
                face.bbox,
                track_lookup,
                threshold=face_in_track_iou,
                dilate_px=dilate_track_px,
                frame_size=(width, height),
            )
            if assigned_track is None:
                continue
            if not _passes_area(face.bbox, frame_area, pipeline_cfg):
                continue
            state = accumulator.active.get(assigned_track)
            if state is None:
                continue
            should_embed = True
            if embed_every_n and embed_every_n > 1:
                if frame_idx - state.last_embed_frame < embed_every_n:
                    should_embed = False
            if should_embed:
                aligned = face_detector.align_to_112(frame, face.landmarks, face.bbox)
                embedding = embedder.embed(aligned)
                match = matcher.best_match(embedding)
                if match is not None:
                    track_similarity[assigned_track] = match[1]
                associate.apply_embedding(state, embedding, matcher, frame_idx)
                state.last_embed_frame = frame_idx
                if state.label:
                    score = track_similarity.get(assigned_track, 0.0)
                    state.add_label(frame_idx, state.label, score)
                    if len(state.label_scores) == 1:
                        LOGGER.debug(
                            "Track %d initial label=%s score=%.3f frame=%d votes=%s",
                            assigned_track,
                            state.label,
                            score,
                            frame_idx,
                            {k: round(v, 3) for k, v in state.label_votes.items()},
                        )
            for box in frame_boxes:
                if box["track_id"] == assigned_track:
                    box["label"] = state.label
                    box["score"] = track_similarity.get(assigned_track)

        if frame_boxes:
            frame_annotations[frame_idx] = frame_boxes

        log_progress(frames_seen)
        last_frames_seen = frames_seen

    cap.release()
    tracks = accumulator.flush()

    if last_frames_seen:
        if last_frames_seen < next_progress_log:
            next_progress_log = last_frames_seen
        log_progress(last_frames_seen)

    initial_labelled_frames = sum(len(t.label_scores) for t in tracks)
    LOGGER.info("Tracks with label_scores (pre-backfill): %d frames total", initial_labelled_frames)

    annotations_by_track: Dict[int, List[Tuple[int, Optional[str], Optional[float]]]] = {}
    for frame_idx, boxes in frame_annotations.items():
        for box in boxes:
            track_id = box.get("track_id")
            if track_id is None:
                continue
            label = box.get("label")
            if not label or label == "UNKNOWN":
                continue
            score = box.get("score")
            annotations_by_track.setdefault(track_id, []).append((frame_idx, label, score))

    backfilled_total = 0
    for track in tracks:
        events = annotations_by_track.get(track.track_id)
        if not events:
            continue
        added = 0
        for frame_idx, label, score in sorted(events, key=lambda item: item[0]):
            before = len(track.label_scores)
            track.add_label(frame_idx, label, score)
            if len(track.label_scores) > before:
                added += 1
        if added:
            LOGGER.info(
                "Track %d: backfilled %d label assignments from frame_annotations",
                track.track_id,
                added,
            )
            backfilled_total += added

    if backfilled_total:
        LOGGER.info("Backfilled %d label assignments from frame_annotations", backfilled_total)

    labelled_frames_total = sum(len(t.label_scores) for t in tracks)
    LOGGER.info("Tracks with label_scores (post-backfill): %d frames total", labelled_frames_total)

    pre_finalize_with_labels = sum(1 for t in tracks if t.label_scores)
    current_subtrack_count = sum(1 for t in tracks if t.current_subtrack_label)
    LOGGER.info(
        "Pre-finalize: tracks with label_scores=%d current_subtrack_label=%d",
        pre_finalize_with_labels,
        current_subtrack_count,
    )

    # Finalize tracks
    for track in tracks:
        associate.finalize_track_subtracks(track, matcher)
        if track.label is None:
            track.label = associate.finalize_label(track)

    tracks_with_scores = [t for t in tracks if t.label_scores]
    subtrack_total = sum(len(t.subtracks) for t in tracks_with_scores)
    LOGGER.info(
        "Finalize summary: %d tracks with label_scores, total subtracks=%d",
        len(tracks_with_scores),
        subtrack_total,
    )
    if tracks_with_scores and subtrack_total == 0:
        LOGGER.warning("Finalize sanity check failed: label_scores present but no subtracks generated")

    labelled_tracks = [t for t in tracks if t.label]
    tracks_with_subtracks = [t for t in tracks if t.subtracks]
    LOGGER.info(
        "Track summary: total=%d labelled=%d subtracks=%d labelled_frames=%d",
        len(tracks),
        len(labelled_tracks),
        sum(len(t.subtracks) for t in tracks_with_subtracks),
        sum(len(t.label_scores) for t in tracks),
    )

    video_stem = infer_video_stem(video_path)
    output_dir = ensure_dir(args.output_dir / video_stem)

    max_gap_ms = args.max_gap_ms if args.max_gap_ms is not None else pipeline_cfg.get("max_gap_ms", 200)
    min_run_ms = args.min_run_ms if args.min_run_ms is not None else pipeline_cfg.get("min_run_ms", 750)

    segments = aggregate.tracks_to_segments(
        tracks,
        fps=fps,
        max_gap_ms=max_gap_ms,
        min_run_ms=min_run_ms,
        use_subtracks=identity_split_enabled,
    )
    
    # Log segment statistics
    labeled_segments = [s for s in segments if s.label and s.label != "UNKNOWN"]
    unlabeled_segments = [s for s in segments if not s.label or s.label == "UNKNOWN"]
    LOGGER.info(
        "Generated %d segments (%d labeled, %d unlabeled) from %d tracks",
        len(segments),
        len(labeled_segments),
        len(unlabeled_segments),
        len(tracks),
    )
    if labeled_segments:
        label_counts = {}
        for seg in labeled_segments:
            label_counts[seg.label] = label_counts.get(seg.label, 0) + 1
        LOGGER.info("Segments by label: %s", label_counts)
    
    totals_df = aggregate.segments_to_totals(segments)
    video_duration_ms = (frame_idx + 1) / fps * 1000.0 if frame_idx >= 0 else 0.0
    timeline_df = aggregate.segments_to_timeline(segments, fps, video_duration_ms)
    track_timeline_df = aggregate.segments_to_track_timeline(segments, video_duration_ms)

    segments_df = (
        pd.DataFrame([seg.to_dict() for seg in segments])
        if segments
        else pd.DataFrame(columns=["byte_track_id", "subtrack_id", "label", "start_ms", "end_ms", "duration_ms", "frames", "avg_similarity"])
    )

    totals_csv = output_dir / f"{video_stem}-TOTALS.csv"
    totals_json = output_dir / f"{video_stem}-TOTALS.json"
    segments_csv = output_dir / f"{video_stem}-segments.csv"
    tracks_csv = output_dir / f"{video_stem}-tracks.csv"
    timeline_csv = output_dir / f"{video_stem}-timeline.csv"
    track_timeline_csv = output_dir / f"{video_stem}-track_timeline.csv"
    annotations_json = output_dir / f"{video_stem}-annotations.json"

    totals_df.to_csv(totals_csv, index=False)
    totals_df.to_json(totals_json, orient="records", indent=2)
    segments_df.to_csv(segments_csv, index=False)
    segments_df.to_csv(tracks_csv, index=False)  # tracks.csv is an alias of segments.csv
    timeline_df.to_csv(timeline_csv, index=False)
    track_timeline_df.to_csv(track_timeline_csv, index=False)
    try:
        persisted_segments_df = pd.read_csv(segments_csv)
        if len(segments) != len(persisted_segments_df):
            LOGGER.warning(
                "Consistency check: generated %d segments but segments.csv has %d rows",
                len(segments),
                len(persisted_segments_df),
            )
        else:
            LOGGER.info("✓ segments.csv rowcount matches generated segments (%d)", len(segments))
    except Exception as exc:
        LOGGER.warning("Unable to re-read segments.csv for validation: %s", exc)

    # Prepare track_segments for annotations
    track_segments = []
    for seg in segments:
        track_segments.append({
            "byte_track_id": seg.byte_track_id,
            "subtrack_id": seg.subtrack_id,
            "label": seg.label,
            "start_frame": int(seg.start_ms / 1000.0 * fps),
            "end_frame": int(seg.end_ms / 1000.0 * fps),
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "duration_ms": seg.duration_ms,
            "frames": seg.frames,
            "avg_similarity": seg.avg_similarity,
        })

    dump_json(
        annotations_json,
        {
            "video": str(video_path),
            "fps": fps,
            "track_segments": track_segments,
            "frame_annotations": [
                {"frame_idx": idx, "tracks": boxes}
                for idx, boxes in sorted(frame_annotations.items())
            ],
        },
    )

    LOGGER.info(
        "Tracker outputs written: totals=%s segments=%s tracks=%s timeline=%s track_timeline=%s annotations=%s",
        totals_csv,
        segments_csv,
        tracks_csv,
        timeline_csv,
        track_timeline_csv,
        annotations_json,
    )

    # Output consistency checks (log only)
    LOGGER.info("Running output consistency checks...")

    # Check 1: Sum of per-label segment durations should match TOTALS.csv
    segment_totals_by_label = {}
    for seg in segments:
        if seg.label and seg.label != "UNKNOWN":
            segment_totals_by_label[seg.label] = segment_totals_by_label.get(seg.label, 0.0) + seg.duration_ms

    totals_by_label = {}
    for _, row in totals_df.iterrows():
        totals_by_label[row["label"]] = row["duration_ms"]

    for label in set(list(segment_totals_by_label.keys()) + list(totals_by_label.keys())):
        seg_total = segment_totals_by_label.get(label, 0.0)
        csv_total = totals_by_label.get(label, 0.0)
        diff = abs(seg_total - csv_total)
        if diff > 1.0:  # Allow 1ms tolerance for rounding
            LOGGER.warning(
                "Consistency check: Label %s segment sum (%.1fms) differs from TOTALS.csv (%.1fms) by %.1fms",
                label, seg_total, csv_total, diff
            )
        else:
            LOGGER.info("✓ Label %s: segment sum matches TOTALS.csv (%.1fms)", label, seg_total)

    # Check 2: For any byte_track_id with labeled frames, there should be >= 1 row in segments.csv
    labeled_track_ids = set()
    for track in tracks:
        if track.label or track.subtracks:
            labeled_track_ids.add(track.track_id)

    segment_track_ids = set(seg.byte_track_id for seg in segments)

    missing_tracks = labeled_track_ids - segment_track_ids
    if missing_tracks:
        LOGGER.warning(
            "Consistency check: %d labeled tracks missing from segments.csv: %s",
            len(missing_tracks),
            sorted(missing_tracks)
        )
    else:
        LOGGER.info("✓ All labeled tracks appear in segments.csv")

    LOGGER.info("Output consistency checks complete.")

    _log_label_coverage(segments, video_duration_ms=video_duration_ms)


def _configure_runtime_threads() -> None:
    """Limit thread usage for embedded runtimes."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")


def _default_providers_for_platform() -> Tuple[str, ...]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def _resolve_providers(
    cli_providers: Optional[Sequence[str]],
    config_providers: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    if cli_providers:
        return tuple(cli_providers)
    if config_providers:
        if isinstance(config_providers, str):  # type: ignore[unreachable]
            return (config_providers,)
        return tuple(config_providers)
    return _default_providers_for_platform()


def _log_label_coverage(segments: Sequence[aggregate.Segment], video_duration_ms: float) -> None:
    if video_duration_ms <= 0:
        LOGGER.info("Coverage report skipped (video duration <= 0)")
        return
    total_seconds = video_duration_ms / 1000.0
    coverage: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        label = seg.label or "UNKNOWN"
        info = coverage.setdefault(label, {"duration_ms": 0.0, "count": 0})
        info["duration_ms"] += max(seg.duration_ms, 0.0)
        info["count"] += 1
    if not coverage:
        LOGGER.info("Coverage report: no labeled segments")
        return
    LOGGER.info("Coverage report (per label):")
    for label, data in sorted(coverage.items(), key=lambda item: item[1]["duration_ms"], reverse=True):
        seconds = data["duration_ms"] / 1000.0
        pct = (seconds / total_seconds * 100.0) if total_seconds > 0 else 0.0
        LOGGER.info("  %s → %.1fs (%.1f%% of video) across %d segments", label, seconds, pct, int(data["count"]))


def _dilate_bbox(
    bbox: Tuple[float, float, float, float],
    dilation: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float, float, float]:
    if dilation <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    dx = width * dilation
    dy = height * dilation
    nx1 = max(0.0, x1 - dx)
    ny1 = max(0.0, y1 - dy)
    nx2 = min(float(frame_width), x2 + dx)
    ny2 = min(float(frame_height), y2 + dy)
    return (nx1, ny1, nx2, ny2)


def _associate_face(
    face_bbox,
    track_lookup,
    threshold: float,
    dilate_px: float,
    frame_size: Tuple[int, int],
) -> Optional[int]:
    best_id: Optional[int] = None
    best_iou = 0.0
    frame_width, frame_height = frame_size
    for track_id, obs in track_lookup.items():
        dilated = _dilate_bbox(obs.bbox, dilate_px, frame_width, frame_height)
        overlap = iou(face_bbox, dilated)
        if overlap >= threshold and overlap > best_iou:
            best_id = track_id
            best_iou = overlap
    return best_id


def _passes_area(face_bbox, frame_area: float, config: Dict) -> bool:
    area = bbox_area(face_bbox)
    min_area_px = config.get("min_area_px")
    if min_area_px and area < min_area_px:
        return False
    min_area_frac = config.get("min_area_frac", 0.0)
    return (area / frame_area) >= min_area_frac


if __name__ == "__main__":
    main()
