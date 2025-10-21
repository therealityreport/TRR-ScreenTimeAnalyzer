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


def _parse_scene_probe_positions(raw: str | Sequence[float] | None) -> tuple[float, ...]:
    """Normalize probe position inputs into a sorted tuple between 0 and 1."""

    if raw is None:
        return (0.1, 0.5, 0.9)

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        items = str(raw).split(",")
        values = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                continue

    probes: list[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            continue
        if not math.isfinite(float(value)):
            continue
        probes.append(float(value))

    if not probes:
        return (0.1, 0.5, 0.9)

    clipped = [min(1.0, max(0.0, probe)) for probe in probes]
    clipped.sort()
    deduped: list[float] = []
    last: Optional[float] = None
    for probe in clipped:
        if last is None or not math.isclose(last, probe, rel_tol=1e-6, abs_tol=1e-6):
            deduped.append(probe)
            last = probe
    return tuple(deduped)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest aligned face crops from a video episode")
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
    parser.set_defaults(scene_aware=None)
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
        help="Disable scene/shot sampling (overrides auto-detection).",
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
        type=int,
        default=3,
        help="How many frames to probe per scene (evenly spaced between 20% and 80%).",
    )
    parser.add_argument(
        "--scene-autoscale-threshold",
        type=float,
        default=180.0,
        help="Automatically enable scene-aware sampling when duration (seconds) is at or below this value (<=0 to disable).",
    )
    parser.add_argument(
        "--scene-probes",
        type=str,
        default="0.1,0.5,0.9",
        help="Comma-separated normalized offsets to probe within a scene (0.0-1.0).",
    )
    parser.add_argument(
        "--scene-probe-interval",
        type=float,
        default=30.0,
        help="Approximate seconds between probes when expanding scene samples (<=0 disables scaling).",
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
    args = parser.parse_args(argv)
    args.scene_probes = _parse_scene_probe_positions(getattr(args, "scene_probes", None))  # type: ignore[assignment]
    return args
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


def _generate_scene_probe_positions(sample_count: int, base_positions: Sequence[float]) -> np.ndarray:
    """Generate normalized probe positions ensuring first/middle/last coverage."""

    if sample_count <= 0:
        return np.array([0.5], dtype=float)

    base = list(base_positions)
    if not base:
        base = [0.1, 0.5, 0.9]
    base = [min(1.0, max(0.0, float(pos))) for pos in base]
    base.sort()

    if len(base) == 1:
        return np.full(sample_count, base[0], dtype=float)

    if sample_count == 1:
        center_idx = min(range(len(base)), key=lambda idx: abs(base[idx] - 0.5))
        return np.array([base[center_idx]], dtype=float)

    if sample_count <= len(base):
        raw_indices = np.linspace(0, len(base) - 1, sample_count)
        chosen: list[int] = []
        last_idx = -1
        max_idx = len(base) - 1
        for raw in raw_indices:
            idx = int(round(raw))
            idx = max(0, min(max_idx, idx))
            if idx <= last_idx and last_idx < max_idx:
                idx = last_idx + 1
            chosen.append(idx)
            last_idx = idx
        return np.array([base[idx] for idx in chosen], dtype=float)

    start = base[0]
    end = base[-1]
    if math.isclose(start, end, rel_tol=1e-6, abs_tol=1e-6):
        return np.full(sample_count, start, dtype=float)
    return np.linspace(start, end, sample_count)


def _scaled_scene_sample_count(
    base_samples: int,
    span_frames: int,
    fps: float,
    interval_seconds: float,
    min_positions: int,
) -> int:
    sample_count = max(1, int(base_samples))
    sample_count = max(sample_count, int(min_positions) if min_positions else 1)
    if fps > 0 and interval_seconds > 0:
        span_seconds = span_frames / fps
        dynamic = int(math.floor(span_seconds / interval_seconds)) + 1
        sample_count = max(sample_count, dynamic)
    return sample_count


def _probe_video_duration(video_path: Path) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """Inspect the input video and return (frames, fps, duration_seconds)."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None

    try:
        frame_count_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    frame_count: Optional[int] = None
    fps: Optional[float] = None
    duration: Optional[float] = None

    if math.isfinite(frame_count_raw) and frame_count_raw > 0:
        frame_count = int(frame_count_raw)

    if math.isfinite(fps_raw) and fps_raw > 0:
        fps = float(fps_raw)

    if frame_count is not None and fps and fps > 0:
        duration = frame_count / fps

    return frame_count, fps, duration


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
    if max_new_tracks_per_sec is None:
        max_new_tracks_per_sec = pipeline_cfg.get("max_new_tracks_per_sec", 2.0)
    max_new_tracks_per_sec = float(max_new_tracks_per_sec)

    pipeline_stitch_default = _parse_bool_flag(pipeline_cfg.get("stitch_identities"), True)
    stitch_identities = _parse_bool_flag(args.stitch_identities, pipeline_stitch_default)

    stitch_sim = args.stitch_sim if args.stitch_sim is not None else pipeline_cfg.get("stitch_sim", 0.45)
    stitch_gap_ms = args.stitch_gap_ms if args.stitch_gap_ms is not None else pipeline_cfg.get("stitch_gap_ms", 8000.0)
    stitch_min_iou = args.stitch_min_iou if args.stitch_min_iou is not None else pipeline_cfg.get("stitch_min_iou", 0.1)
    stitch_sim = float(stitch_sim)
    stitch_gap_ms = float(stitch_gap_ms)
    stitch_min_iou = float(stitch_min_iou)

    identity_guard_stride = int(pipeline_cfg.get("identity_guard_stride", 6))
    identity_guard_consecutive = int(pipeline_cfg.get("identity_guard_consecutive", 3))
    identity_guard_cosine_reject = float(pipeline_cfg.get("identity_guard_cosine_reject", 0.35))
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
        profile_asymmetry_thresh=pipeline_cfg.get("profile_asymmetry_thresh", 0.25),
        quality_weights=quality_weights,
        target_area_frac=target_area_frac,
        debug_rejections=pipeline_cfg.get("debug_rejections", False),
        multi_face_per_track_guard=pipeline_cfg.get("multi_face_per_track_guard", True),
        multi_face_tiebreak=pipeline_cfg.get("multi_face_tiebreak", "quality"),
        fallback_head_pct=float(pipeline_cfg.get("fallback_head_pct", 0.4)),
        identity_guard=identity_guard_enabled,
        identity_split=identity_split_enabled,
        identity_sim_threshold=identity_sim_threshold,
        identity_min_picks=identity_min_picks,
        identity_guard_stride=identity_guard_stride,
        identity_guard_consecutive=identity_guard_consecutive,
        identity_guard_cosine_reject=identity_guard_cosine_reject,
        reindex_harvest_tracks=bool(pipeline_cfg.get("reindex_harvest_tracks", True)),
        fast_mode=args.fast,
        min_track_frames=min_track_frames_value,
        write_candidates=bool(args.write_candidates),
        recall_pass=bool(args.recall_pass),
        recall_det_thresh=0.20,
        recall_face_iou=float(pipeline_cfg.get("recall_face_iou", 0.15)),
        recall_track_iou=float(pipeline_cfg.get("recall_track_iou", 0.30)),
        recall_max_gap=int(pipeline_cfg.get("recall_max_gap", 4)),
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
        total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(total_frames_raw) if math.isfinite(total_frames_raw) and total_frames_raw > 0 else 0
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
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if math.isfinite(fps_raw) and fps_raw > 0 else 0.0

        if args.scene_samples <= 0:
            LOGGER.warning("scene-samples=%d is not positive; defaulting to 1 probe per scene.", args.scene_samples)

        base_sample_count = max(1, int(args.scene_samples))
        min_position_count = len(getattr(args, "scene_probes", ()))
        frames_to_process: list[dict[str, object]] = []
        seen_keys: set[tuple[int, int]] = set()
        probe_lookup: dict[tuple[int, int], dict[str, object]] = {}
        scene_probe_plan: dict[int, dict[str, object]] = {}

        for scene_idx, (start_frame, end_frame) in enumerate(scenes):
            if end_frame < start_frame:
                LOGGER.warning(
                    "Skipping scene %d with invalid range start=%d end=%d.", scene_idx, start_frame, end_frame
                )
                continue
            span = max(1, end_frame - start_frame)
            sample_count = _scaled_scene_sample_count(
                base_sample_count,
                span,
                fps,
                float(getattr(args, "scene_probe_interval", 0.0) or 0.0),
                min_position_count,
            )
            positions = _generate_scene_probe_positions(sample_count, getattr(args, "scene_probes", ()))
            scene_seconds = (span / fps) if fps > 0 else None
            plan_entry = scene_probe_plan.setdefault(
                scene_idx,
                {
                    "start": int(start_frame),
                    "end": int(end_frame),
                    "frames": [],
                    "positions": [],
                    "seconds": scene_seconds,
                    "count": len(positions),
                },
            )
            plan_entry["count"] = len(positions)
            plan_entry["seconds"] = scene_seconds
            plan_entry["positions"] = [float(pos) for pos in positions]
            for probe_idx, pos in enumerate(positions):
                frame_idx = int(round(start_frame + pos * span))
                frame_idx = max(start_frame, min(end_frame, frame_idx))
                if total_frames > 0 and frame_idx >= total_frames:
                    continue
                key = (scene_idx, frame_idx)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                plan_entry.setdefault("frames", []).append(int(frame_idx))
                probe_info = {
                    "scene_idx": scene_idx,
                    "frame_idx": int(frame_idx),
                    "probe_index": int(probe_idx),
                    "probe_position": float(pos),
                    "probe_count": len(positions),
                    "scene_start": int(start_frame),
                    "scene_end": int(end_frame),
                    "scene_span": span,
                    "scene_seconds": scene_seconds,
                }
                frames_to_process.append(probe_info)
                probe_lookup[key] = probe_info

        frames_to_process.sort(key=lambda item: (item["frame_idx"], item["scene_idx"]))

        candidates_csv = out_dir / "candidates.csv"
        candidate_counts: Counter[int] = Counter()
        total_candidates = 0

        ensure_dir(out_dir / "candidates")

        preview_records: list[dict[str, object]] = []

        with candidates_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "scene_idx",
                    "frame_idx",
                    "probe_index",
                    "probe_position",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "conf",
                    "rel_path",
                ]
            )

            selected_records: list[dict[str, object]] = []
            scene_reason = "scene_probe_auto" if getattr(args, "scene_auto_inferred", False) else "scene_probe_manual"

            for probe in frames_to_process:
                scene_idx = int(probe["scene_idx"])
                frame_idx = int(probe["frame_idx"])
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
                            int(probe["probe_index"]),
                            float(probe["probe_position"]),
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
                    probe_meta = probe_lookup.get((scene_idx, frame_idx), probe)
                    selected_records.append(
                        {
                            "track_id": -1,
                            "byte_track_id": -1,
                            "frame": frame_idx,
                            "quality": None,
                            "sharpness": None,
                            "frontalness": None,
                            "area_frac": area_frac,
                            "picked": True,
                            "reason": scene_reason,
                            "path": rel_record_path.as_posix(),
                            "association_iou": None,
                            "match_mode": None,
                            "frame_offset": None,
                            "identity_cosine": None,
                            "similarity_to_centroid": None,
                            "provider": "scene_probe",
                            "scene_idx": scene_idx,
                            "scene_probe_index": int(probe_meta.get("probe_index", probe["probe_index"])),
                            "scene_probe_position": float(probe_meta.get("probe_position", probe["probe_position"])),
                            "scene_probe_count": int(probe_meta.get("probe_count", len(getattr(args, "scene_probes", ())))),
                            "scene_probe_auto": bool(getattr(args, "scene_auto_inferred", False)),
                        }
                    )
                    if args.cluster_preview:
                        preview_records.append(
                            {
                                "scene_idx": scene_idx,
                                "frame_idx": frame_idx,
                                "rel_path": rel_record_path.as_posix(),
                                "abs_path": crop_path,
                                "probe_index": int(probe_meta.get("probe_index", probe["probe_index"])),
                            }
                        )

        selected_csv = out_dir / "selected_samples.csv"
        backup_csv: Optional[Path] = None
        if selected_csv.exists():
            backup_csv = out_dir / "selected_samples.scene_probe_backup.csv"
            try:
                selected_csv.rename(backup_csv)
                LOGGER.warning(
                    "Existing selected_samples.csv moved to %s before writing scene probe summary.",
                    backup_csv,
                )
            except OSError as exc:
                LOGGER.warning("Failed to rotate previous selected_samples.csv: %s", exc)

        fieldnames = [
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
            "scene_idx",
            "scene_probe_index",
            "scene_probe_position",
            "scene_probe_count",
            "scene_probe_auto",
        ]

        with selected_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            if selected_records:
                writer.writerows(selected_records)

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
        if scene_probe_plan:
            for idx, plan in scene_probe_plan.items():
                LOGGER.debug(
                    "Scene %d probes frames=%s positions=%s count=%d span_frames=%d span_seconds=%s auto=%s",
                    idx,
                    plan.get("frames", []),
                    [f"{pos:.3f}" for pos in plan.get("positions", [])],
                    plan.get("count", 0),
                    int(plan.get("end", 0)) - int(plan.get("start", 0)),
                    ("{:.2f}".format(plan["seconds"]) if plan.get("seconds") is not None else "unknown"),
                    bool(getattr(args, "scene_auto_inferred", False)),
                )
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
    args.scene_auto_inferred = False  # type: ignore[attr-defined]
    args.scene_auto_duration = None  # type: ignore[attr-defined]
    args.scene_auto_frame_count = None  # type: ignore[attr-defined]
    args.scene_auto_fps = None  # type: ignore[attr-defined]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video: {args.video}")

    samples_per_sec = args.samples_per_sec
    try:
        target_samples = None if samples_per_sec is None else float(samples_per_sec)
    except (TypeError, ValueError):
        target_samples = None
    if target_samples is not None and target_samples > 0:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
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

    configure_threads(args.threads)
    args.session_options = build_session_options(args.threads)  # type: ignore[attr-defined]
    setup_logging()

    scene_pref = getattr(args, "scene_aware", None)
    autoscale_threshold = float(getattr(args, "scene_autoscale_threshold", 0.0) or 0.0)
    if scene_pref is None and autoscale_threshold > 0:
        frame_count, fps, duration = _probe_video_duration(args.video)
        args.scene_auto_frame_count = frame_count  # type: ignore[attr-defined]
        args.scene_auto_fps = fps  # type: ignore[attr-defined]
        args.scene_auto_duration = duration  # type: ignore[attr-defined]
        if duration is not None and duration <= autoscale_threshold:
            args.scene_aware = True
            args.scene_auto_inferred = True  # type: ignore[attr-defined]
            LOGGER.info(
                "Auto-enabled scene-aware sampling for %s (duration=%.2fs <= threshold=%.2fs).",
                args.video,
                duration,
                autoscale_threshold,
            )
        else:
            args.scene_aware = False
            if duration is None:
                LOGGER.debug(
                    "Unable to determine duration for %s; scene-aware auto-enable skipped (threshold %.2fs).",
                    args.video,
                    autoscale_threshold,
                )
            else:
                LOGGER.debug(
                    "Duration %.2fs exceeds scene-aware threshold %.2fs; using standard harvest.",
                    duration,
                    autoscale_threshold,
                )
    elif scene_pref is not None:
        args.scene_aware = bool(scene_pref)
    else:
        args.scene_aware = False

    if args.scene_aware:
        run_scene_aware_harvest(args)
        return
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
