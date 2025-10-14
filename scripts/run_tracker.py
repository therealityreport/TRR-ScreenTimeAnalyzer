#!/usr/bin/env python3
"""CLI for running YOLO + ByteTrack + ArcFace recognition pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
        help="ONNX execution providers",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    pipeline_cfg = load_yaml(args.pipeline_config)
    tracker_cfg = load_yaml(args.tracker_config)

    det_size = tuple(args.retina_det_size) if args.retina_det_size else tuple(pipeline_cfg.get("det_size", [960, 960]))
    face_conf = args.face_det_threshold or pipeline_cfg.get("face_conf_th", 0.45)
    person_conf = args.person_conf or pipeline_cfg.get("person_conf_th", 0.20)
    stride = pipeline_cfg.get("stride", 1)

    person_detector = YOLOPersonDetector(weights=args.person_weights, conf_thres=person_conf)
    face_detector = RetinaFaceDetector(det_size=det_size, det_thresh=face_conf)
    tracker = ByteTrackWrapper(**tracker_cfg)

    embedder = ArcFaceEmbedder(model_path=args.arcface_model, providers=args.providers)
    facebank_embeddings = load_facebank(args.facebank_parquet)
    matcher = TrackVotingMatcher(
        facebank_embeddings,
        similarity_th=pipeline_cfg.get("similarity_th", 0.82),
        vote_decay=pipeline_cfg.get("vote_decay", 0.99),
        flip_tolerance=pipeline_cfg.get("flip_tolerance", 0.30),
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
            1 for track in accumulator.finished.values() if track.label
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
            assigned_track = _associate_face(face.bbox, track_lookup, pipeline_cfg.get("face_in_track_iou", 0.30))
            if assigned_track is None:
                continue
            if not _passes_area(face.bbox, frame_area, pipeline_cfg):
                continue
            state = accumulator.active.get(assigned_track)
            if state is None:
                continue
            aligned = face_detector.align_to_112(frame, face.landmarks, face.bbox)
            embedding = embedder.embed(aligned)
            match = matcher.best_match(embedding)
            if match is not None:
                track_similarity[assigned_track] = match[1]
            associate.apply_embedding(state, embedding, matcher)

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

    for track in tracks:
        if track.label is None:
            track.label = associate.finalize_label(track)

    video_stem = infer_video_stem(video_path)
    output_dir = ensure_dir(args.output_dir / video_stem)

    segments = aggregate.tracks_to_segments(
        tracks,
        fps=fps,
        max_gap_ms=pipeline_cfg.get("max_gap_ms", 500),
        min_run_ms=pipeline_cfg.get("min_run_ms", 200),
    )
    totals_df = aggregate.segments_to_totals(segments)
    video_duration_ms = (frame_idx + 1) / fps * 1000.0 if frame_idx >= 0 else 0.0
    timeline_df = aggregate.segments_to_timeline(segments, fps, video_duration_ms)

    segments_df = (
        pd.DataFrame([seg.to_dict() for seg in segments])
        if segments
        else pd.DataFrame(columns=["track_id", "label", "start_ms", "end_ms", "duration_ms"])
    )

    totals_csv = output_dir / f"{video_stem}-TOTALS.csv"
    totals_json = output_dir / f"{video_stem}-TOTALS.json"
    segments_csv = output_dir / f"{video_stem}-segments.csv"
    timeline_csv = output_dir / f"{video_stem}-timeline.csv"
    annotations_json = output_dir / f"{video_stem}-annotations.json"

    totals_df.to_csv(totals_csv, index=False)
    totals_df.to_json(totals_json, orient="records", indent=2)
    segments_df.to_csv(segments_csv, index=False)
    timeline_df.to_csv(timeline_csv, index=False)

    dump_json(
        annotations_json,
        {
            "video": str(video_path),
            "fps": fps,
            "frame_annotations": [
                {"frame_idx": idx, "tracks": boxes}
                for idx, boxes in sorted(frame_annotations.items())
            ],
        },
    )

    LOGGER.info(
        "Tracker outputs written: totals=%s segments=%s timeline=%s annotations=%s",
        totals_csv,
        segments_csv,
        timeline_csv,
        annotations_json,
    )


def _associate_face(face_bbox, track_lookup, threshold: float) -> Optional[int]:
    best_id: Optional[int] = None
    best_iou = 0.0
    for track_id, obs in track_lookup.items():
        overlap = iou(face_bbox, obs.bbox)
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
