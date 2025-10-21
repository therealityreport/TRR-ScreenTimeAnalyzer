#!/usr/bin/env python3
"""CLI for harvesting aligned face crops and manifests."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import Counter
import shutil
from distutils.util import strtobool
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import pandas as pd

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.harvest.harvest import HarvestConfig, HarvestRunner
from screentime.io_utils import ensure_dir, infer_video_stem, load_yaml, setup_logging
from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.segmentation.scenes import detect_scenes
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper


LOGGER = logging.getLogger("scripts.harvest")
CPU_PROVIDER_NAME = "CPUExecutionProvider"


def _providers_cpu_only(providers: Optional[Sequence[str]]) -> bool:
    if not providers:
        return True
    normalized = [str(provider).strip() for provider in providers if provider]
    if not normalized:
        return True
    return all(provider.upper() == CPU_PROVIDER_NAME.upper() for provider in normalized)


def _maybe_apply_cpu_preset(args: argparse.Namespace, pipeline_cfg: dict) -> bool:
    if not (
        _providers_cpu_only(getattr(args, "onnx_providers", None))
        and not getattr(args, "fast", False)
        and getattr(args, "retina_det_size", None) is None
    ):
        args._cpu_preset = False  # type: ignore[attr-defined]
        return False

    default_stride = pipeline_cfg.get("stride", 1)
    try:
        stride_value = int(default_stride)
    except (TypeError, ValueError):
        stride_value = 1
    min_stride = max(2, stride_value)

    current_stride = getattr(args, "stride", None)
    try:
        current_stride_int = None if current_stride is None else int(current_stride)
    except (TypeError, ValueError):
        current_stride_int = None

    if current_stride_int is None or current_stride_int < min_stride:
        args.stride = min_stride
    else:
        args.stride = current_stride_int

    args._cpu_preset = True  # type: ignore[attr-defined]
    return True


def _normalize_threads(value: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _parse_bool_flag(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    try:
        return bool(strtobool(str(value)))
    except (ValueError, AttributeError):
        return default


def _resolve_min_frontalness(args: argparse.Namespace, pipeline_cfg: dict, legacy_default: float = 0.20) -> float:
    """Resolve the minimum frontalness threshold honoring overrides."""

    cli_override = None
    if getattr(args, "frontalness_thresh", None) is not None:
        cli_override = args.frontalness_thresh
    elif getattr(args, "min_frontalness", None) is not None:
        cli_override = args.min_frontalness

    if cli_override is not None:
        try:
            return float(cli_override)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid CLI frontalness override %r; falling back to config/default", cli_override)

    cfg_value = pipeline_cfg.get("min_frontalness")
    if cfg_value is not None:
        try:
            return float(cfg_value)
        except (TypeError, ValueError):
            LOGGER.warning("Invalid pipeline min_frontalness %r; falling back to legacy default", cfg_value)

    return float(legacy_default)


class _TrackExplicitInt(argparse.Action):
    """Argparse action that records when a numeric value was explicitly provided."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            value = int(values)
        except (TypeError, ValueError):
            parser.error(f"{option_string} expects an integer value")
        setattr(namespace, self.dest, value)
        setattr(namespace, f"_{self.dest}_explicit", True)


def _compute_scene_positions(sample_count: int, custom: Optional[Sequence[float]] = None) -> np.ndarray:
    """Resolve normalized probe positions for scene-aware sampling."""

    if custom:
        positions = []
        for pos in custom:
            try:
                positions.append(float(pos))
            except (TypeError, ValueError):
                continue
        if positions:
            clamped = [min(0.99, max(0.0, p)) for p in positions]
            return np.array(sorted(clamped), dtype=float)

    count = max(1, int(sample_count))
    if count == 1:
        return np.array([0.5], dtype=float)
    if count == 2:
        return np.array([0.1, 0.9], dtype=float)

    base = [0.05, 0.5, 0.95]
    if count <= 3:
        return np.array(base[:count], dtype=float)

    remaining = count - len(base)
    if remaining > 0:
        interior = np.linspace(0.2, 0.8, remaining + 2)[1:-1]
        positions = base + interior.tolist()
    else:
        positions = base
    positions = positions[:count]
    return np.array(sorted(positions), dtype=float)


def configure_threads(thread_count: int) -> None:
    """Limit math library thread usage to manage thermals."""
    threads = _normalize_threads(thread_count)
    defaults = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "ORT_INTRA_OP_NUM_THREADS": str(threads),
        "ORT_INTER_OP_NUM_THREADS": "1",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)
    try:
        import torch  # type: ignore

        torch.set_num_threads(threads)
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        cv2.setNumThreads(threads)
    except Exception:  # pragma: no cover - optional backend
        pass


def build_session_options(thread_count: int):
    """Create ONNX Runtime session options honoring thread caps."""
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None
    threads = _normalize_threads(thread_count)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = threads
    session_options.inter_op_num_threads = 1
    return session_options


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest aligned face crops from a video episode")
    parser.set_defaults(scene_aware=None)
    parser.add_argument("video", type=Path, help="Path to the input video file")
    parser.add_argument(
        "--harvest-dir",
        type=Path,
        default=None,
        help="Directory to store harvested crops (no nested dataset folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # backward compatibility shim handled post-parse
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
        default=None,
        help="Path to YOLO person detector weights (required unless --scene-aware is set)",
    )
    parser.add_argument(
        "--retina-det-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Override RetinaFace detection size (CPU preset uses 640x640 when unset).",
    )
    parser.add_argument(
        "--face-det-threshold",
        type=float,
        default=None,
        help="Override face detection confidence threshold",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=None,
        help="Alias for --face-det-threshold (defaults to 0.30 when unset).",
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
        help="Override number of samples to keep per track (default 8)",
    )
    parser.add_argument(
        "--min-gap-frames",
        type=int,
        default=None,
        help="Override temporal gap between selected samples",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=None,
        help="Minimum ByteTrack frames required to keep a track (default 12).",
    )
    parser.add_argument(
        "--new-track-min-frames",
        type=int,
        default=None,
        help="Frames required before confirming a new track (default 3).",
    )
    parser.add_argument(
        "--max-new-tracks-per-sec",
        type=float,
        default=None,
        help="Maximum number of new tracks to confirm per second (default 2).",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=None,
        help="Override ByteTrack track_buffer size in frames (default from tracker config).",
    )
    parser.add_argument(
        "--stitch-identities",
        type=lambda value: bool(strtobool(value)),
        default=None,
        metavar="{true,false}",
        help="Enable post-harvest identity stitching (default true).",
    )
    parser.add_argument(
        "--stitch-sim",
        type=float,
        default=None,
        help="Cosine distance threshold for stitching clusters (default 0.45).",
    )
    parser.add_argument(
        "--stitch-gap-ms",
        type=float,
        default=None,
        help="Maximum allowed gap in milliseconds when stitching (default 8000).",
    )
    parser.add_argument(
        "--stitch-min-iou",
        type=float,
        default=None,
        help="Minimum IoU for stitching adjacency (default 0.1).",
    )
    parser.add_argument(
        "--min-frontalness",
        type=float,
        default=None,
        help="Override minimum frontalness required for selection eligibility",
    )
    parser.add_argument(
        "--frontalness-thresh",
        type=float,
        default=None,
        help="Alias for --min-frontalness (defaults to pipeline config value, or 0.20 if unspecified).",
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
        help="Enable fast mode (stride=2, 640 detector, threads=1, disable debug rejects).",
    )
    parser.add_argument(
        "--samples-per-sec",
        type=float,
        default=None,
        help="Target samples per second. When set, overrides --stride by sampling ceil(FPS / value).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override base frame stride before harvest sampling (CPU preset uses max(2, pipeline stride)).",
    )
    parser.add_argument(
        "--onnx-providers",
        type=str,
        nargs="+",
        default=None,
        help="ONNX execution providers (leave unset or CPUExecutionProvider to trigger CPU preset; e.g., 'CoreMLExecutionProvider CPUExecutionProvider')",
    )
    parser.add_argument(
        "--defer-embeddings",
        action="store_true",
        help="Skip ArcFace embeddings during harvest (compute later in clustering/facebank stages).",
    )
    parser.add_argument(
        "--no-identity-guard",
        action="store_true",
        help="Disable identity purity guard and split checks (faster but risks mixing identities).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Maximum compute threads for math/ONNX libraries (default: 1).",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Progress log cadence as a percentage of frames processed (default: 1%%).",
    )
    parser.add_argument(
        "--heartbeat-sec",
        type=float,
        default=2.0,
        help="Emit a heartbeat log at this interval in seconds (default: 2s).",
    )
    parser.add_argument(
        "--scene-aware",
        dest="scene_aware",
        action="store_true",
        help="Use scene/shot detection and sample a few frames per scene.",
    )
    parser.add_argument(
        "--no-scene-aware",
        dest="scene_aware",
        action="store_false",
        help="Force disable scene-aware sampling even when auto-detection would enable it.",
    )
    parser.add_argument(
        "--no-auto-scene-aware",
        action="store_true",
        help="Disable automatic scene-aware activation for short clips.",
    )
    parser.add_argument(
        "--scene-auto-threshold-sec",
        type=float,
        default=300.0,
        help="Automatically enable scene-aware mode when clip duration is at or below this threshold (seconds).",
    )
    parser.add_argument("--scene-threshold", type=float, default=27.0, help="Scene detection threshold.")
    parser.add_argument(
        "--scene-min-frames",
        type=int,
        default=12,
        help="Minimum scene length in frames for the content detector.",
    )
    parser.add_argument(
        "--scene-samples",
        action=_TrackExplicitInt,
        default=3,
        help="How many frames to probe per scene (evenly spaced between 20% and 80%).",
    )
    parser.add_argument(
        "--scene-probe-positions",
        type=float,
        nargs="+",
        default=None,
        help="Custom normalized positions (0-1) for scene-aware probes; overrides the default staggered pattern.",
    )
    parser.add_argument(
        "--min-face-frac",
        type=float,
        default=0.02,
        help="Discard faces smaller than this fraction of the frame area.",
    )
    parser.add_argument(
        "--min-face",
        type=int,
        default=48,
        help="Minimum face bbox size in pixels for selection (shorter side, default 48).",
    )
    parser.add_argument(
        "--cluster-preview",
        action="store_true",
        help="Generate unsupervised identity clusters for harvested candidates (scene-aware mode only).",
    )
    parser.add_argument(
        "--cluster-eps",
        type=float,
        default=0.35,
        help="Cosine distance epsilon for DBSCAN when clustering candidates.",
    )
    parser.add_argument(
        "--cluster-min-samples",
        type=int,
        default=2,
        help="Minimum samples per cluster for DBSCAN when running --cluster-preview.",
    )
    parser.add_argument(
        "--write-candidates",
        action="store_true",
        help="Persist intermediate candidate crops under candidates/ (default disabled).",
    )
    parser.add_argument(
        "--recall-pass",
        dest="recall_pass",
        action="store_true",
        default=True,
        help="Enable relaxed recall pass to backfill missed detections (default on).",
    )
    parser.add_argument(
        "--no-recall-pass",
        dest="recall_pass",
        action="store_false",
        help="Disable the relaxed recall pass.",
    )
    parser.add_argument(
        "--recall-det-thresh",
        type=float,
        default=None,
        help="Override detection confidence threshold for recall candidates (default 0.20).",
    )
    parser.add_argument(
        "--recall-face-iou",
        type=float,
        default=None,
        help="Override minimum face-to-track IoU required for recall promotion.",
    )
    parser.add_argument(
        "--recall-track-iou",
        type=float,
        default=None,
        help="Override minimum dilated track IoU required for recall promotion.",
    )
    return parser.parse_args(argv)

def _resolve_existing_path(path_str: str, nested_dir: Path) -> Path:
    candidate = Path(path_str)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((Path.cwd() / candidate).resolve())
    candidates.append((nested_dir / candidate).resolve())
    for item in candidates:
        if item.exists():
            return item
    return candidates[0]


def _relativize_to_harvest(path_str: str, nested_dir: Path, harvest_dir: Path) -> str:
    abs_path = _resolve_existing_path(path_str, nested_dir).resolve()
    try:
        return abs_path.relative_to(harvest_dir).as_posix()
    except ValueError:
        pass
    try:
        return abs_path.relative_to(nested_dir).as_posix()
    except ValueError:
        pass
    parts = list(abs_path.parts)
    for idx, part in enumerate(parts):
        if part.startswith("track_"):
            return Path(*parts[idx:]).as_posix()
    return abs_path.name


def migrate_nested_harvest(out_root: Path) -> None:
    if not out_root.exists():
        return
    nested_dirs = [p for p in out_root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    for nested in nested_dirs:
        LOGGER.info("Flattening nested harvest layout under %s", nested)
        manifest_path = nested / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for entry in data or []:
                for sample in entry.get("samples", []):
                    path_str = sample.get("path")
                    if path_str:
                        sample["path"] = _relativize_to_harvest(path_str, nested, out_root)
            target_manifest = out_root / "manifest.json"
            target_manifest.parent.mkdir(parents=True, exist_ok=True)
            with target_manifest.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            manifest_path.unlink(missing_ok=True)
        selected_csv = nested / "selected_samples.csv"
        if selected_csv.exists():
            with selected_csv.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            for row in rows:
                path_str = row.get("path") or ""
                if path_str:
                    row["path"] = _relativize_to_harvest(path_str, nested, out_root)
                else:
                    row["path"] = ""
            target_csv = out_root / "selected_samples.csv"
            target_csv.parent.mkdir(parents=True, exist_ok=True)
            with target_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            selected_csv.unlink(missing_ok=True)
        for item in list(nested.iterdir()):
            target = out_root / item.name
            if target.exists():
                LOGGER.debug("Skipping existing target during migration: %s", target)
                continue
            shutil.move(str(item), str(target))
        try:
            nested.rmdir()
        except OSError:
            LOGGER.debug("Nested directory not empty after migration: %s", nested)



def run_standard_harvest(args: argparse.Namespace, cap: Optional[cv2.VideoCapture] = None) -> None:
    pipeline_cfg = load_yaml(args.pipeline_config)
    tracker_cfg = load_yaml(args.tracker_config)
    cpu_preset_enabled = _maybe_apply_cpu_preset(args, pipeline_cfg)

    fps_hint = float(getattr(args, "_video_fps", 0.0) or 0.0)
    auto_scene = bool(getattr(args, "_auto_scene_enabled", False))

    track_buffer_cfg = tracker_cfg.get("track_buffer")
    track_buffer_val = None
    if args.track_buffer is not None:
        track_buffer_val = max(1, int(args.track_buffer))
        tracker_cfg["track_buffer"] = track_buffer_val
        LOGGER.info("Applying CLI track_buffer override: %d", track_buffer_val)
    elif fps_hint > 0:
        base_buffer = int(track_buffer_cfg) if isinstance(track_buffer_cfg, (int, float)) else 30
        buffer_seconds = float(pipeline_cfg.get("track_buffer_seconds", 1.0))
        adaptive_buffer = max(base_buffer, int(math.ceil(fps_hint * buffer_seconds)))
        if adaptive_buffer > base_buffer:
            tracker_cfg["track_buffer"] = adaptive_buffer
            track_buffer_val = adaptive_buffer
            LOGGER.info(
                "Scaling track_buffer to %d based on %.2ffps clip (window %.2fs).",
                adaptive_buffer,
                fps_hint,
                buffer_seconds,
            )
    if track_buffer_val is None and isinstance(track_buffer_cfg, (int, float)):
        track_buffer_val = int(track_buffer_cfg)

    video_stem = infer_video_stem(args.video)
    legacy_layout = args.harvest_dir is None
    if args.harvest_dir is not None:
        output_root = Path(args.harvest_dir)
        harvest_dir = output_root
    elif args.output_dir is not None:
        output_root = Path(args.output_dir)
        harvest_dir = output_root / video_stem
    else:
        output_root = Path("data/harvest")
        harvest_dir = output_root / video_stem
    output_root = output_root.expanduser().resolve()
    harvest_dir = harvest_dir.expanduser().resolve()
    ensure_dir(harvest_dir)
    migrate_nested_harvest(harvest_dir)

    det_size_cfg = pipeline_cfg.get("det_size", [960, 960]) or [960, 960]
    det_size = tuple(args.retina_det_size) if args.retina_det_size else tuple(det_size_cfg)
    det_thresh_arg = args.det_thresh if args.det_thresh is not None else args.face_det_threshold
    face_conf = float(det_thresh_arg) if det_thresh_arg is not None else 0.30
    person_conf = float(args.person_conf) if args.person_conf is not None else float(pipeline_cfg.get("person_conf_th", 0.20))

    if cpu_preset_enabled and args.retina_det_size is None:
        det_size = (640, 640)
    elif args.fast and args.retina_det_size is None:
        det_size = (640, 640)
    if getattr(args, "_cpu_preset", False):
        stride_for_log = args.stride if args.stride is not None else pipeline_cfg.get("stride", 1)
        LOGGER.info(
            "Applying CPU preset: stride=%s, detector=%dx%d. Override with --stride or --retina-det-size.",
            stride_for_log,
            det_size[0],
            det_size[1],
        )

    person_detector = YOLOPersonDetector(
        weights=args.person_weights,
        conf_thres=person_conf,
    )

    providers = tuple(args.onnx_providers) if args.onnx_providers else None
    face_detector = RetinaFaceDetector(
        det_size=det_size,
        det_thresh=face_conf,
        providers=providers,
        threads=int(args.threads),
        session_options=getattr(args, "session_options", None),
        user_det_size_override=getattr(args, "_retina_size_override", False),
    )
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
    min_frontalness = _resolve_min_frontalness(args, pipeline_cfg)
    profile_quota = int(pipeline_cfg.get("profile_quota", 0))
    profile_min_frontal = float(pipeline_cfg.get("profile_min_frontalness", 0.0))
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
    if hasattr(args, "min_area_frac") and args.min_area_frac is not None:
        min_area_frac = float(args.min_area_frac)
    frontal_pctile = pipeline_cfg.get("frontal_pctile")
    if frontal_pctile is not None:
        frontal_pctile = float(frontal_pctile)
    min_frontal_picks = int(pipeline_cfg.get("min_frontal_picks", 2))
    min_face_px = int(max(0, args.min_face))
    progress_fallback_frames = max(1, int(pipeline_cfg.get("progress_fallback_frames", 500)))
    min_track_frames_value = args.min_track_frames
    if min_track_frames_value is None:
        min_track_frames_value = pipeline_cfg.get("min_track_frames", 12)
    min_track_frames_value = int(min_track_frames_value)

    new_track_min_frames = args.new_track_min_frames
    if new_track_min_frames is None:
        new_track_min_frames = pipeline_cfg.get("new_track_min_frames", 3)
    new_track_min_frames = int(new_track_min_frames)

    max_new_tracks_per_sec = args.max_new_tracks_per_sec
    max_tracks_from_cfg = "max_new_tracks_per_sec" in pipeline_cfg
    if max_new_tracks_per_sec is None:
        max_new_tracks_per_sec = pipeline_cfg.get("max_new_tracks_per_sec", 2.0)
        if not max_tracks_from_cfg and fps_hint > 0:
            adaptive = max(2.0, round(fps_hint / 12.0, 2))
            if auto_scene:
                adaptive = max(adaptive, round(fps_hint / 8.0, 2))
            if adaptive > float(max_new_tracks_per_sec):
                LOGGER.info(
                    "Scaling max_new_tracks_per_sec to %.2f for %.2ffps clip (auto_scene=%s).",
                    adaptive,
                    fps_hint,
                    auto_scene,
                )
                max_new_tracks_per_sec = adaptive
    max_new_tracks_per_sec = float(max_new_tracks_per_sec)

    pipeline_stitch_default = _parse_bool_flag(pipeline_cfg.get("stitch_identities"), True)
    stitch_identities = _parse_bool_flag(args.stitch_identities, pipeline_stitch_default)

    stitch_sim = args.stitch_sim if args.stitch_sim is not None else pipeline_cfg.get("stitch_sim", 0.45)
    stitch_gap_ms = args.stitch_gap_ms if args.stitch_gap_ms is not None else pipeline_cfg.get("stitch_gap_ms", 8000.0)
    stitch_min_iou = args.stitch_min_iou if args.stitch_min_iou is not None else pipeline_cfg.get("stitch_min_iou", 0.1)
    stitch_sim = float(stitch_sim)
    stitch_gap_ms = float(stitch_gap_ms)
    stitch_min_iou = float(stitch_min_iou)

    recall_det_thresh = args.recall_det_thresh
    if recall_det_thresh is None:
        recall_det_thresh = pipeline_cfg.get("recall_det_thresh", 0.20)
    recall_face_iou = args.recall_face_iou
    if recall_face_iou is None:
        recall_face_iou = pipeline_cfg.get("recall_face_iou", 0.15)
    recall_track_iou = args.recall_track_iou
    if recall_track_iou is None:
        recall_track_iou = pipeline_cfg.get("recall_track_iou", 0.30)
    recall_det_thresh = float(recall_det_thresh)
    recall_face_iou = float(recall_face_iou)
    recall_track_iou = float(recall_track_iou)

    identity_guard_stride = int(pipeline_cfg.get("identity_guard_stride", 6))
    identity_guard_consecutive = int(pipeline_cfg.get("identity_guard_consecutive", 3))
    identity_guard_cosine_reject = float(pipeline_cfg.get("identity_guard_cosine_reject", 0.35))
    identity_guard_recovery = int(pipeline_cfg.get("identity_guard_recovery", HarvestConfig.identity_guard_recovery))
    identity_guard_recover_margin = float(
        pipeline_cfg.get("identity_guard_recover_margin", HarvestConfig.identity_guard_recover_margin)
    )
    identity_guard_enabled = _parse_bool_flag(pipeline_cfg.get("identity_guard"), True)
    identity_split_enabled = _parse_bool_flag(pipeline_cfg.get("identity_split"), True)
    if getattr(args, "no_identity_guard", False):
        LOGGER.warning(
            "Identity purity checks disabled via --no-identity-guard; guard and split enforcement will be skipped."
        )
        identity_guard_enabled = False
        identity_split_enabled = False
    identity_sim_threshold = float(pipeline_cfg.get("identity_sim_threshold", 0.55))
    identity_min_picks = int(pipeline_cfg.get("identity_min_picks", 3))

    harvest_config = HarvestConfig(
        stride=pipeline_cfg.get("stride", 1),
        samples_min=pipeline_cfg.get("samples_min", 4),
        samples_max=pipeline_cfg.get("samples_max", 12),
        samples_per_track=samples_per_track,
        min_gap_frames=min_gap_frames,
        min_area_frac=min_area_frac,
        min_area_px=pipeline_cfg.get("min_area_px"),
        min_face_px=min_face_px,
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
        profile_asymmetry_thresh=float(
            pipeline_cfg.get("profile_asymmetry_thresh", HarvestConfig.profile_asymmetry_thresh)
        ),
        profile_quota=profile_quota,
        profile_min_frontalness=profile_min_frontal,
        quality_weights=quality_weights,
        target_area_frac=target_area_frac,
        debug_rejections=bool(pipeline_cfg.get("debug_rejections", False)),
        multi_face_per_track_guard=bool(pipeline_cfg.get("multi_face_per_track_guard", True)),
        multi_face_tiebreak=pipeline_cfg.get("multi_face_tiebreak", "quality"),
        fallback_head_pct=float(pipeline_cfg.get("fallback_head_pct", 0.4)),
        identity_guard=identity_guard_enabled,
        identity_split=identity_split_enabled,
        identity_sim_threshold=identity_sim_threshold,
        identity_min_picks=identity_min_picks,
        identity_guard_stride=identity_guard_stride,
        identity_guard_consecutive=identity_guard_consecutive,
        identity_guard_cosine_reject=identity_guard_cosine_reject,
        identity_guard_recovery=identity_guard_recovery,
        identity_guard_recover_margin=identity_guard_recover_margin,
        reindex_harvest_tracks=bool(pipeline_cfg.get("reindex_harvest_tracks", True)),
        fast_mode=args.fast,
        min_track_frames=min_track_frames_value,
        write_candidates=bool(args.write_candidates),
        recall_pass=bool(args.recall_pass),
        recall_det_thresh=recall_det_thresh,
        recall_face_iou=recall_face_iou,
        recall_track_iou=recall_track_iou,
        recall_max_gap=int(pipeline_cfg.get("recall_max_gap", HarvestConfig.recall_max_gap)),
        progress_percent_interval=float(args.progress_interval),
        progress_fallback_frames=progress_fallback_frames,
        heartbeat_seconds=float(args.heartbeat_sec),
        defer_embeddings=bool(args.defer_embeddings),
        threads=int(args.threads),
        new_track_min_frames=new_track_min_frames,
        max_new_tracks_per_sec=max_new_tracks_per_sec,
        stitch_identities=stitch_identities,
        stitch_sim=stitch_sim,
        stitch_gap_ms=stitch_gap_ms,
        stitch_min_iou=stitch_min_iou,
    )

    if args.stride is not None:
        harvest_config.stride = max(1, int(args.stride))
    elif args.fast:
        harvest_config.stride = max(2, harvest_config.stride)

    runner = HarvestRunner(person_detector, face_detector, tracker, harvest_config)
    capture = cap if cap is not None else getattr(args, "_capture", None)
    manifest_path = runner.run(args.video, output_root, legacy_layout=legacy_layout, cap=capture)
    LOGGER.info("Harvest manifest written to %s", manifest_path)

def run_scene_aware_harvest(args: argparse.Namespace, cap: Optional[cv2.VideoCapture] = None) -> None:
    pipeline_cfg = load_yaml(args.pipeline_config)
    cpu_preset_enabled = _maybe_apply_cpu_preset(args, pipeline_cfg)
    det_size_cfg = pipeline_cfg.get("det_size", [960, 960]) or [960, 960]
    det_size = tuple(args.retina_det_size) if args.retina_det_size else tuple(det_size_cfg)
    face_conf = args.face_det_threshold or pipeline_cfg.get("face_conf_th", 0.45)

    if cpu_preset_enabled and args.retina_det_size is None:
        det_size = (640, 640)
    elif args.fast and args.retina_det_size is None:
        det_size = (640, 640)
    if getattr(args, "_cpu_preset", False):
        stride_for_log = args.stride if args.stride is not None else pipeline_cfg.get("stride", 1)
        LOGGER.info(
            "Applying CPU preset: stride=%s, detector=%dx%d. Override with --stride or --retina-det-size.",
            stride_for_log,
            det_size[0],
            det_size[1],
        )

    providers = tuple(args.onnx_providers) if args.onnx_providers else None
    face_detector = RetinaFaceDetector(
        det_size=det_size,
        det_thresh=face_conf,
        providers=providers,
        threads=int(args.threads),
        session_options=getattr(args, "session_options", None),
        user_det_size_override=getattr(args, "_retina_size_override", False),
    )

    video_stem = infer_video_stem(args.video)
    if args.harvest_dir is not None:
        output_root = Path(args.harvest_dir)
        harvest_dir = output_root
    elif args.output_dir is not None:
        output_root = Path(args.output_dir)
        harvest_dir = output_root / video_stem
    else:
        output_root = Path("data/harvest")
        harvest_dir = output_root / video_stem
    output_root = output_root.expanduser().resolve()
    out_dir = ensure_dir(harvest_dir.expanduser().resolve())
    migrate_nested_harvest(out_dir)

    capture = cap if cap is not None else getattr(args, "_capture", None)
    owns_cap = False
    if capture is None:
        capture = cv2.VideoCapture(str(args.video))
        owns_cap = True
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            LOGGER.warning("Video %s reported zero frame count; seek accuracy may be limited.", args.video)

        scenes = detect_scenes(
            str(args.video),
            threshold=args.scene_threshold,
            min_scene_len=args.scene_min_frames,
        )
        if not scenes:
            if total_frames > 0:
                fallback_scene = (0, max(0, total_frames - 1))
                scenes = [fallback_scene]
                LOGGER.warning(
                    "Scene detection found no segments. Falling back to single scene covering frames %s.",
                    fallback_scene,
                )
            else:
                LOGGER.warning(
                    "Scene detection found no segments and video length is unavailable; no probes will be sampled."
                )

        scenes_csv = out_dir / "scenes.csv"
        with scenes_csv.open("w", encoding="utf-8") as fh:
            fh.write("scene_idx,start_frame,end_frame\n")
            for idx, (start, end) in enumerate(scenes):
                fh.write(f"{idx},{start},{end}\n")
        LOGGER.info("Wrote %s with %d scenes.", scenes_csv, len(scenes))

        sample_count = max(1, args.scene_samples)
        if args.scene_samples <= 0:
            LOGGER.warning("scene-samples=%d is not positive; defaulting to 1 probe per scene.", args.scene_samples)

        if getattr(args, "_auto_scene_enabled", False) and not getattr(args, "_scene_samples_explicit", False):
            base = sample_count
            sample_count = max(sample_count, 5)
            if sample_count != base:
                LOGGER.info(
                    "Auto scene-aware: increasing probes per scene from %d to %d for richer coverage.",
                    base,
                    sample_count,
                )

        positions = _compute_scene_positions(sample_count, args.scene_probe_positions)

        frames_to_process: list[tuple[int, int]] = []
        seen_keys: set[tuple[int, int]] = set()
        for scene_idx, (start_frame, end_frame) in enumerate(scenes):
            if end_frame < start_frame:
                LOGGER.warning(
                    "Skipping scene %d with invalid range start=%d end=%d.", scene_idx, start_frame, end_frame
                )
                continue
            span = max(1, end_frame - start_frame)
            for pos in positions:
                frame_idx = int(round(start_frame + pos * span))
                frame_idx = max(start_frame, min(end_frame, frame_idx))
                if total_frames > 0 and frame_idx >= total_frames:
                    continue
                key = (scene_idx, frame_idx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                frames_to_process.append(key)

        frames_to_process.sort(key=lambda item: (item[1], item[0]))

        candidates_csv = out_dir / "candidates.csv"
        candidate_counts: Counter[int] = Counter()
        total_candidates = 0

        ensure_dir(out_dir / "candidates")

        preview_records: list[dict[str, object]] = []

        with candidates_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["scene_idx", "frame_idx", "x1", "y1", "x2", "y2", "conf", "rel_path"])

            for scene_idx, frame_idx in frames_to_process:
                if frame_idx < 0:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = capture.read()
                if not ok or frame is None:
                    LOGGER.warning("Failed to read frame %d for scene %d.", frame_idx, scene_idx)
                    continue

                height, width = frame.shape[:2]
                frame_area = float(height * width)
                if frame_area <= 0:
                    LOGGER.debug("Frame %d has zero area; skipping.", frame_idx)
                    continue

                detections = face_detector.detect(frame, frame_idx)
                for det_idx, det in enumerate(detections):
                    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
                    if x2 <= x1 or y2 <= y1:
                        continue

                    area_frac = ((x2 - x1) * (y2 - y1)) / frame_area
                    if area_frac < args.min_face_frac:
                        continue

                    x1c = max(0, x1)
                    y1c = max(0, y1)
                    x2c = min(width, x2)
                    y2c = min(height, y2)
                    if x2c <= x1c or y2c <= y1c:
                        continue

                    crop = frame[y1c:y2c, x1c:x2c]
                    rel_dir = ensure_dir(
                        out_dir / "candidates" / (f"scene_{scene_idx:04d}" if scene_idx >= 0 else "noscenes")
                    )
                    rel_name = f"F{frame_idx:06d}_C{det_idx:02d}.jpg"
                    crop_path = rel_dir / rel_name
                    if not cv2.imwrite(str(crop_path), crop):
                        LOGGER.warning("Failed to write crop %s", crop_path)
                        continue

                    rel_record_path = crop_path.relative_to(out_dir)
                    writer.writerow(
                        [
                            scene_idx,
                            frame_idx,
                            x1,
                            y1,
                            x2,
                            y2,
                            float(getattr(det, "score", 1.0)),
                            rel_record_path.as_posix(),
                        ]
                    )
                    candidate_counts[scene_idx] += 1
                    total_candidates += 1
                    if args.cluster_preview:
                        preview_records.append(
                            {
                                "scene_idx": scene_idx,
                                "frame_idx": frame_idx,
                                "rel_path": rel_record_path.as_posix(),
                                "abs_path": crop_path,
                            }
                        )

        scene_count = len(scenes)
        LOGGER.info(
            "Scene-aware harvest complete: %d scenes, %d probe frames, %d candidate crops.",
            scene_count,
            len(frames_to_process),
            total_candidates,
        )
        if scene_count:
            distribution = [candidate_counts.get(idx, 0) for idx in range(scene_count)]
            LOGGER.info("Candidates per scene: %s", distribution)
        else:
            LOGGER.info("No scenes detected; candidates per scene not available.")

        if args.cluster_preview and preview_records:
            run_cluster_preview(
                out_dir=out_dir,
                preview_records=preview_records,
                eps=max(1e-6, float(args.cluster_eps)),
                min_samples=max(1, int(args.cluster_min_samples)),
                provider_overrides=args.onnx_providers,
                threads=int(args.threads),
            )
        elif args.cluster_preview:
            LOGGER.warning("Cluster preview requested but no candidate crops were written; skipping clustering.")
    finally:
        if owns_cap:
            capture.release()


def run_cluster_preview(
    out_dir: Path,
    preview_records: list[dict[str, object]],
    eps: float,
    min_samples: int,
    provider_overrides: Optional[list[str]],
    threads: int,
) -> None:
    """Embed harvested candidates and cluster by cosine similarity for quick review."""
    LOGGER.info("Running cluster preview on %d candidate crops.", len(preview_records))
    from sklearn.cluster import DBSCAN

    provider_list: Optional[tuple[str, ...]] = None
    if provider_overrides:
        providers = list(provider_overrides)
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        provider_list = tuple(providers)

    embedder = ArcFaceEmbedder(providers=provider_list, threads=threads)
    embeddings: list[np.ndarray] = []
    kept_records: list[dict[str, object]] = []

    for record in preview_records:
        crop_path = Path(record["abs_path"])
        image = cv2.imread(str(crop_path))
        if image is None:
            LOGGER.warning("Cluster preview skipping unreadable crop %s", crop_path)
            continue
        aligned = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
        try:
            embedding = embedder.embed(aligned)
        except Exception as exc:  # pragma: no cover - inference errors surface in logs
            LOGGER.warning("Cluster preview failed to embed %s: %s", crop_path, exc)
            continue
        embeddings.append(embedding)
        kept_records.append(record)

    if not embeddings:
        LOGGER.warning("Cluster preview aborted: no embeddings could be computed.")
        return

    embedding_matrix = np.stack(embeddings, axis=0)
    LOGGER.info(
        "Embedding %d crops with DBSCAN (eps=%.3f, min_samples=%d, metric=cosine).",
        embedding_matrix.shape[0],
        eps,
        min_samples,
    )
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embedding_matrix)

    clusters_dir = ensure_dir(out_dir / "clusters")
    assignments: list[dict[str, object]] = []
    per_cluster_counts: Counter[int] = Counter()

    unique_labels = sorted(set(int(lbl) for lbl in labels))
    track_id_map: dict[int, int] = {label: label for label in unique_labels if label >= 0}
    next_track_id = (max(track_id_map.values()) + 1) if track_id_map else 0
    noise_next = next_track_id

    for idx, (record, label) in enumerate(zip(kept_records, labels)):
        label = int(label)
        per_cluster_counts[label] += 1
        cluster_name = f"cluster_{label:04d}" if label >= 0 else "noise"
        cluster_path = ensure_dir(clusters_dir / cluster_name)
        src_path = Path(record["abs_path"])
        dst_path = cluster_path / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as exc:  # pragma: no cover - filesystem errors
            LOGGER.warning("Cluster preview failed to copy %s -> %s: %s", src_path, dst_path, exc)
            continue
        if label >= 0:
            track_id = track_id_map.setdefault(label, label)
        else:
            track_id = noise_next
            noise_next += 1
        assignments.append(
            {
                "cluster_id": label,
                "track_id": int(track_id),
                "scene_idx": int(record["scene_idx"]),
                "frame_idx": int(record["frame_idx"]),
                "rel_path": str(record["rel_path"]),
                "cluster_rel_path": dst_path.relative_to(out_dir).as_posix(),
                "cluster_path": str(dst_path),
                "label": "",
            }
        )

    if not assignments:
        LOGGER.warning("Cluster preview created no assignments; check earlier warnings.")
        return

    clusters_parquet = out_dir / "clusters.parquet"
    assignments_df = pd.DataFrame(assignments)
    assignments_df.to_parquet(clusters_parquet, index=False)

    samples_records: list[dict[str, object]] = []
    for row in assignments_df.itertuples():
        samples_records.append(
            {
                "track_id": int(row.track_id),
                "byte_track_id": int(row.track_id),
                "frame": int(row.frame_idx),
                "quality": None,
                "sharpness": None,
                "frontalness": None,
                "area_frac": None,
                "picked": True,
                "reason": "clustered",
                "path": str((out_dir / row.cluster_rel_path).resolve()),
                "association_iou": None,
                "match_mode": None,
                "frame_offset": None,
                "identity_cosine": None,
                "similarity_to_centroid": None,
                "provider": "cluster_preview",
                "is_debug": False,
                "cluster_id": int(row.cluster_id),
                "scene_idx": int(row.scene_idx),
            }
        )

    samples_df = pd.DataFrame(samples_records)
    selected_cols = [
        "track_id",
        "byte_track_id",
        "frame",
        "quality",
        "sharpness",
        "frontalness",
        "area_frac",
        "picked",
        "reason",
        "path",
        "association_iou",
        "match_mode",
        "frame_offset",
        "identity_cosine",
        "similarity_to_centroid",
        "provider",
        "is_debug",
        "cluster_id",
        "scene_idx",
    ]
    samples_df = samples_df[selected_cols]
    selected_samples_path = out_dir / "selected_samples.csv"
    backup_path = None
    if selected_samples_path.exists():
        backup_path = out_dir / "selected_samples.scene_backup.csv"
        samples_df.to_csv(backup_path, index=False)
        LOGGER.warning(
            "selected_samples.csv already exists; wrote scene cluster samples to %s instead.", backup_path
        )
    else:
        samples_df.to_csv(selected_samples_path, index=False)

    manifest_entries: list[dict[str, object]] = []
    for track_id, group in assignments_df.groupby("track_id"):
        manifest_entries.append(
            {
                "track_id": int(track_id),
                "byte_track_id": int(track_id),
                "sample_count": int(len(group)),
                "cluster_id": int(group["cluster_id"].iloc[0]),
                "scene_indices": sorted({int(idx) for idx in group["scene_idx"]}),
                "min_frame_idx": int(group["frame_idx"].min()),
                "max_frame_idx": int(group["frame_idx"].max()),
            }
        )

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest_backup = out_dir / "manifest.scene_backup.json"
        with manifest_backup.open("w", encoding="utf-8") as fh:
            json.dump(manifest_entries, fh, indent=2)
        LOGGER.warning("manifest.json already exists; wrote scene cluster manifest to %s.", manifest_backup)
    else:
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest_entries, fh, indent=2)

    cluster_summary = {
        ("cluster_{}".format(label) if label >= 0 else "noise"): count
        for label, count in sorted(per_cluster_counts.items(), key=lambda kv: kv[0])
    }
    LOGGER.info(
        "Cluster preview complete: %d assignments across %d clusters (including noise). Outputs: %s, %s",
        len(assignments),
        len(cluster_summary),
        clusters_parquet,
        selected_samples_path if backup_path is None else backup_path,
    )
    LOGGER.info("Cluster distribution: %s", cluster_summary)


def main() -> None:
    args = parse_args()
    threads_explicit = args.threads is not None
    args.threads = _normalize_threads(args.threads or 1)
    args.progress_interval = max(0.1, float(args.progress_interval))
    args.heartbeat_sec = max(0.1, float(args.heartbeat_sec))
    args._retina_size_override = args.retina_det_size is not None  # type: ignore[attr-defined]
    args._cpu_preset = False  # type: ignore[attr-defined]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = 0.0
    if fps > 0 and total_frames > 0:
        duration_sec = total_frames / fps

    args._video_fps = fps  # type: ignore[attr-defined]
    args._video_frame_count = total_frames  # type: ignore[attr-defined]
    args._video_duration_sec = duration_sec if duration_sec > 0 else None  # type: ignore[attr-defined]

    auto_scene_enabled = False
    if args.scene_aware is None:
        enable_auto = not getattr(args, "no_auto_scene_aware", False)
        if enable_auto and duration_sec > 0 and duration_sec <= float(args.scene_auto_threshold_sec):
            args.scene_aware = True
            auto_scene_enabled = True
            LOGGER.info(
                "Auto-enabling scene-aware harvest for %.1fs clip (threshold %.1fs).",
                duration_sec,
                float(args.scene_auto_threshold_sec),
            )
        else:
            args.scene_aware = False
    setattr(args, "_auto_scene_enabled", auto_scene_enabled)

    samples_per_sec = args.samples_per_sec
    try:
        target_samples = None if samples_per_sec is None else float(samples_per_sec)
    except (TypeError, ValueError):
        target_samples = None
    if target_samples is not None and target_samples > 0:
        if fps > 0.0 and getattr(args, "stride", None) is None:
            args.stride = max(1, int(math.ceil(fps / target_samples)))

    args._capture = cap  # type: ignore[attr-defined]

    if args.fast:
        if args.retina_det_size is None:
            args.retina_det_size = [640, 640]
        try:
            stride_int = None if args.stride is None else int(args.stride)
        except (TypeError, ValueError):
            stride_int = None
        if stride_int is None or stride_int < 2:
            args.stride = 2
        else:
            args.stride = stride_int
        if not threads_explicit:
            args.threads = 1

    try:
        configure_threads(args.threads)
        args.session_options = build_session_options(args.threads)  # type: ignore[attr-defined]
        setup_logging()

        if args.scene_aware:
            run_scene_aware_harvest(args, cap=cap)
        else:
            if not args.person_weights:
                raise SystemExit("--person-weights is required unless --scene-aware is enabled.")

            run_standard_harvest(args, cap=cap)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
