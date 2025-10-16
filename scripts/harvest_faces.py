#!/usr/bin/env python3
"""CLI for harvesting aligned face crops and manifests."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
import shutil
from pathlib import Path

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
        default=None,
        help="Path to YOLO person detector weights (required unless --scene-aware is set)",
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
    parser.add_argument(
        "--onnx-providers",
        type=str,
        nargs="+",
        default=None,
        help="ONNX execution providers (e.g., 'CPUExecutionProvider' or 'CoreMLExecutionProvider CPUExecutionProvider')",
    )
    parser.add_argument(
        "--scene-aware",
        action="store_true",
        help="Use scene/shot detection and sample a few frames per scene.",
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
        "--min-face-frac",
        type=float,
        default=0.02,
        help="Discard faces smaller than this fraction of the frame area.",
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
    return parser.parse_args()


def run_standard_harvest(args: argparse.Namespace) -> None:
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
    
    # Set ONNX providers for RetinaFace
    providers = None
    if args.onnx_providers:
        providers = tuple(args.onnx_providers)
    
    face_detector = RetinaFaceDetector(
        det_size=det_size, 
        det_thresh=face_conf,
        providers=providers
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


def run_scene_aware_harvest(args: argparse.Namespace) -> None:
    pipeline_cfg = load_yaml(args.pipeline_config)
    det_size = tuple(args.retina_det_size) if args.retina_det_size else tuple(pipeline_cfg.get("det_size", [960, 960]))
    face_conf = args.face_det_threshold or pipeline_cfg.get("face_conf_th", 0.45)

    if args.fast and args.retina_det_size is None:
        det_size = (640, 640)

    providers = tuple(args.onnx_providers) if args.onnx_providers else None
    face_detector = RetinaFaceDetector(det_size=det_size, det_thresh=face_conf, providers=providers)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

        out_root = ensure_dir(args.output_dir)
        video_stem = infer_video_stem(args.video)
        out_dir = ensure_dir(out_root / video_stem)

        scenes_csv = out_dir / "scenes.csv"
        with scenes_csv.open("w", encoding="utf-8") as fh:
            fh.write("scene_idx,start_frame,end_frame\n")
            for idx, (start, end) in enumerate(scenes):
                fh.write(f"{idx},{start},{end}\n")
        LOGGER.info("Wrote %s with %d scenes.", scenes_csv, len(scenes))

        sample_count = max(1, args.scene_samples)
        if args.scene_samples <= 0:
            LOGGER.warning("scene-samples=%d is not positive; defaulting to 1 probe per scene.", args.scene_samples)

        if sample_count == 1:
            positions = np.array([0.5], dtype=float)
        else:
            positions = np.linspace(0.2, 0.8, sample_count)

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
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
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
            )
        elif args.cluster_preview:
            LOGGER.warning("Cluster preview requested but no candidate crops were written; skipping clustering.")
    finally:
        cap.release()


def run_cluster_preview(
    out_dir: Path,
    preview_records: list[dict[str, object]],
    eps: float,
    min_samples: int,
    provider_overrides: Optional[list[str]],
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

    embedder = ArcFaceEmbedder(providers=provider_list)
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
    setup_logging()

    if args.scene_aware:
        run_scene_aware_harvest(args)
        return

    if not args.person_weights:
        raise SystemExit("--person-weights is required unless --scene-aware is enabled.")

    run_standard_harvest(args)


if __name__ == "__main__":
    main()
