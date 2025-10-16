#!/usr/bin/env python3
"""CLI for harvesting aligned face crops and manifests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.harvest.harvest import HarvestConfig, HarvestRunner
from screentime.io_utils import ensure_dir, load_yaml, setup_logging
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper


LOGGER = logging.getLogger("scripts.harvest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest aligned face crops from a video episode")
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/harvest"),
        help="Directory to store harvested crops",
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path("configs/bytetrack.yaml"),
        help="Path to ByteTrack configuration YAML",
    )
    parser.add_argument(
        "--person-weights",
        type=str,
        required=True,
        help="Path to YOLO person detector weights",
    )
    parser.add_argument(
        "--retina-det-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Override RetinaFace detection size",
    )
    parser.add_argument(
        "--face-det-threshold",
        type=float,
        default=None,
        help="Override face detection confidence threshold",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=None,
        help="Override person detection confidence threshold",
    )
    parser.add_argument(
        "--quality-weights",
        type=float,
        nargs=3,
        metavar=("SHARPNESS", "FRONTALNESS", "AREA"),
        default=None,
        help="Override quality weighting (sharpness, frontalness, area)",
    )
    parser.add_argument(
        "--target-area-frac",
        type=float,
        default=None,
        help="Override desired face area fraction",
    )
    parser.add_argument(
        "--min-sharpness-laplacian",
        type=float,
        default=None,
        help="Override minimum Laplacian sharpness threshold",
    )
    parser.add_argument(
        "--face-in-track-iou",
        type=float,
        default=None,
        help="Override minimum IoU between face and track",
    )
    parser.add_argument(
        "--samples-per-track",
        type=int,
        default=None,
        help="Override number of samples to keep per track",
    )
    parser.add_argument(
        "--min-gap-frames",
        type=int,
        default=None,
        help="Override temporal gap between selected samples",
    )
    parser.add_argument(
        "--min-frontalness",
        type=float,
        default=None,
        help="Override minimum frontalness required for selection eligibility",
    )
    parser.add_argument(
        "--min-sharpness-pct",
        type=float,
        default=None,
        help="Override per-track Laplacian percentile gate for selection",
    )
    parser.add_argument(
        "--dilate-track-px",
        type=float,
        default=None,
        help="Override person-box dilation factor before IoU",
    )
    parser.add_argument(
        "--temporal-iou-tolerance",
        type=int,
        default=None,
        help="Override temporal tolerance (+/- frames) when matching face to track",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode (stride=2, 640 detector, no debug rejects)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    pipeline_cfg = load_yaml(args.pipeline_config)
    tracker_cfg = load_yaml(args.tracker_config)

    det_size = tuple(args.retina_det_size) if args.retina_det_size else tuple(pipeline_cfg.get("det_size", [960, 960]))
    face_conf = args.face_det_threshold or pipeline_cfg.get("face_conf_th", 0.45)
    person_conf = args.person_conf or pipeline_cfg.get("person_conf_th", 0.20)

    if args.fast and args.retina_det_size is None:
        det_size = (640, 640)

    person_detector = YOLOPersonDetector(
        weights=args.person_weights,
        conf_thres=person_conf,
    )
    face_detector = RetinaFaceDetector(det_size=det_size, det_thresh=face_conf)
    tracker = ByteTrackWrapper(**tracker_cfg)

    quality_weights_cfg = args.quality_weights or pipeline_cfg.get("quality_weights", (0.5, 0.3, 0.2))
    quality_weights = tuple(float(x) for x in quality_weights_cfg)
    target_area_frac = args.target_area_frac or float(pipeline_cfg.get("target_area_frac", 0.02))
    min_sharpness = args.min_sharpness_laplacian
    if min_sharpness is None:
        cfg_min_sharpness = pipeline_cfg.get("min_sharpness_laplacian")
        min_sharpness = float(cfg_min_sharpness) if cfg_min_sharpness is not None else None
    face_in_track_iou = args.face_in_track_iou or float(pipeline_cfg.get("face_in_track_iou", 0.25))
    samples_per_track = args.samples_per_track or int(pipeline_cfg.get("samples_per_track", 8))
    min_gap_frames = args.min_gap_frames or int(pipeline_cfg.get("min_gap_frames", 8))
    min_frontalness = args.min_frontalness or float(pipeline_cfg.get("min_frontalness", 0.35))
    sharpness_pctile = pipeline_cfg.get("sharpness_pctile")
    if sharpness_pctile is None:
        legacy_pct = pipeline_cfg.get("min_sharpness_pct")
        sharpness_pctile = float(legacy_pct) if legacy_pct is not None else None
    min_sharpness_pct = args.min_sharpness_pct
    if min_sharpness_pct is None and sharpness_pctile is None:
        cfg_pct = pipeline_cfg.get("min_sharpness_pct")
        min_sharpness_pct = float(cfg_pct) if cfg_pct is not None else None
    dilate_track_px = args.dilate_track_px or float(pipeline_cfg.get("dilate_track_px", 0.07))
    temporal_iou_tolerance = args.temporal_iou_tolerance or int(pipeline_cfg.get("temporal_iou_tolerance", 1))
    min_area_frac = float(pipeline_cfg.get("min_area_frac", 0.005))
    if hasattr(args, "min_area_frac") and args.min_area_frac is not None:  # future-proof
        min_area_frac = args.min_area_frac
    frontal_pctile = pipeline_cfg.get("frontal_pctile")
    if frontal_pctile is not None:
        frontal_pctile = float(frontal_pctile)
    min_frontal_picks = int(pipeline_cfg.get("min_frontal_picks", 2))

    harvest_config = HarvestConfig(
        stride=pipeline_cfg.get("stride", 1),
        samples_min=pipeline_cfg.get("samples_min", 4),
        samples_max=pipeline_cfg.get("samples_max", 12),
        samples_per_track=samples_per_track,
        min_gap_frames=min_gap_frames,
        min_area_frac=min_area_frac,
        min_area_px=pipeline_cfg.get("min_area_px"),
        min_sharpness_laplacian=min_sharpness,
        min_sharpness_pct=min_sharpness_pct,
        sharpness_pctile=sharpness_pctile,
        min_frontalness=min_frontalness,
        frontal_pctile=frontal_pctile,
        min_frontal_picks=min_frontal_picks,
        face_in_track_iou=face_in_track_iou,
        allow_face_center=bool(pipeline_cfg.get("allow_face_center", False)),
        dilate_track_px=dilate_track_px,
        temporal_iou_tolerance=temporal_iou_tolerance,
        profile_asymmetry_thresh=pipeline_cfg.get("profile_asymmetry_thresh", 0.25),
        quality_weights=quality_weights,
        target_area_frac=target_area_frac,
        debug_rejections=pipeline_cfg.get("debug_rejections", False),
        multi_face_per_track_guard=pipeline_cfg.get("multi_face_per_track_guard", True),
        multi_face_tiebreak=pipeline_cfg.get("multi_face_tiebreak", "quality"),
        fallback_head_pct=float(pipeline_cfg.get("fallback_head_pct", 0.4)),
        identity_guard=bool(pipeline_cfg.get("identity_guard", True)),
        identity_split=bool(pipeline_cfg.get("identity_split", True)),
        identity_sim_threshold=float(pipeline_cfg.get("identity_sim_threshold", 0.62)),
        identity_min_picks=int(pipeline_cfg.get("identity_min_picks", 3)),
        reindex_harvest_tracks=bool(pipeline_cfg.get("reindex_harvest_tracks", True)),
        fast_mode=args.fast,
    )

    if args.fast:
        harvest_config.stride = max(2, harvest_config.stride)
        harvest_config.debug_rejections = False


    runner = HarvestRunner(person_detector, face_detector, tracker, harvest_config)
    ensure_dir(args.output_dir)
    manifest_path = runner.run(args.video, args.output_dir)
    LOGGER.info("Harvest manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
