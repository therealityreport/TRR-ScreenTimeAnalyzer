#!/usr/bin/env python3
"""
Diagnostic script to visualize harvest face detection and rejection reasons.

Usage:
    python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --frames 100-150
    python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --sample 50
"""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.io_utils import load_yaml, setup_logging
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper, TrackAccumulator
from screentime.types import bbox_area, iou

LOGGER = logging.getLogger("scripts.diagnose_harvest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose face detection and harvest issues")
    parser.add_argument("--video", type=Path, required=True, help="Video file to analyze")
    parser.add_argument(
        "--frames",
        type=str,
        help="Frame range to analyze (e.g., '100-200'). If not specified, uses --sample"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of frames to sample evenly throughout video (default: 50)"
    )
    parser.add_argument("--output", type=Path, default=Path("diagnostics"), help="Output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pipeline.yaml"),
        help="Pipeline config file"
    )
    parser.add_argument(
        "--person-weights",
        type=str,
        default="models/weights/yolov8n.pt",
        help="YOLO person detector weights"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames to disk"
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=10,
        help="When saving frames, save every Nth frame (default: 10)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    
    config = load_yaml(args.config)
    args.output.mkdir(exist_ok=True, parents=True)
    
    # Initialize detectors
    LOGGER.info("Initializing detectors...")
    person_det = YOLOPersonDetector(
        weights=args.person_weights,
        conf_thres=config.get("person_conf_th", 0.20)
    )
    face_det = RetinaFaceDetector(
        det_size=tuple(config.get("det_size", [960, 960])),
        det_thresh=config.get("face_conf_th", 0.45)
    )
    
    tracker_cfg = load_yaml(Path("configs/bytetrack.yaml"))
    tracker = ByteTrackWrapper(**tracker_cfg)
    accumulator = TrackAccumulator()
    
    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {args.video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_area = width * height
    
    tracker.set_frame_rate(fps)
    
    # Determine which frames to analyze
    if args.frames:
        start, end = map(int, args.frames.split('-'))
        frames_to_analyze = list(range(start, min(end + 1, total_frames)))
    else:
        # Sample evenly throughout video
        step = max(1, total_frames // args.sample)
        frames_to_analyze = list(range(0, total_frames, step))[:args.sample]
    
    LOGGER.info("="*60)
    LOGGER.info("DIAGNOSTIC CONFIGURATION")
    LOGGER.info("="*60)
    LOGGER.info("Video: %s", args.video.name)
    LOGGER.info("Resolution: %dx%d", width, height)
    LOGGER.info("FPS: %.2f", fps)
    LOGGER.info("Total frames: %d", total_frames)
    LOGGER.info("Analyzing %d frames", len(frames_to_analyze))
    LOGGER.info("")
    LOGGER.info("Detection thresholds:")
    LOGGER.info("  Face confidence: %.2f", config.get("face_conf_th", 0.45))
    LOGGER.info("  Person confidence: %.2f", config.get("person_conf_th", 0.20))
    LOGGER.info("  Min area fraction: %.4f (%.0f pixels)", 
                config.get("min_area_frac", 0.005),
                config.get("min_area_frac", 0.005) * frame_area)
    LOGGER.info("  Face-in-track IoU: %.2f", config.get("face_in_track_iou", 0.25))
    LOGGER.info("  Track dilation: %.2f", config.get("dilate_track_px", 0.07))
    LOGGER.info("="*60)
    
    rejection_stats = Counter()
    detection_log: List[Dict] = []
    association_log: List[Dict] = []
    frames_analyzed = 0
    
    frame_idx = -1
    frames_to_analyze_set = set(frames_to_analyze)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        if frame_idx not in frames_to_analyze_set:
            continue
        
        frames_analyzed += 1
        timestamp_ms = (frame_idx / fps) * 1000.0
        
        # Detect persons and track
        person_dets = person_det.detect(frame, frame_idx)
        observations = tracker.update(person_dets, frame.shape)
        accumulator.update(frame_idx, timestamp_ms, observations)
        
        # Detect faces
        face_dets = face_det.detect(frame, frame_idx)
        
        LOGGER.info("Frame %d: %d persons, %d faces", frame_idx, len(observations), len(face_dets))
        
        # Analyze each face
        for face_idx, face in enumerate(face_dets):
            x1, y1, x2, y2 = face.bbox
            area = bbox_area(face.bbox)
            area_frac = area / frame_area
            
            # Check rejection reasons
            rejection_reasons = []
            
            if area_frac < config.get("min_area_frac", 0.005):
                rejection_reasons.append("area_too_small")
            
            if face.score < config.get("face_conf_th", 0.45):
                rejection_reasons.append("low_confidence")
            
            # Check association with tracks
            best_track_id = None
            best_iou = 0.0
            dilate_px = config.get("dilate_track_px", 0.07)
            iou_thresh = config.get("face_in_track_iou", 0.25)
            
            for obs in observations:
                # Dilate person box
                px1, py1, px2, py2 = obs.bbox
                pw = px2 - px1
                ph = py2 - py1
                dx = pw * dilate_px
                dy = ph * dilate_px
                dilated = (
                    max(0.0, px1 - dx),
                    max(0.0, py1 - dy),
                    min(float(width), px2 + dx),
                    min(float(height), py2 + dy)
                )
                
                overlap = iou(face.bbox, dilated)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = obs.track_id
            
            associated = best_iou >= iou_thresh
            if not associated and len(observations) > 0:
                rejection_reasons.append("no_track_association")
            
            # Record stats
            if rejection_reasons:
                for reason in rejection_reasons:
                    rejection_stats[reason] += 1
                rejection_stats["total_rejected"] += 1
            else:
                rejection_stats["accepted"] += 1
            
            # Log detection
            detection_log.append({
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "face_id": face_idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area_px": int(area),
                "area_pct": area_frac * 100,
                "confidence": face.score,
                "track_id": best_track_id,
                "track_iou": best_iou,
                "associated": associated,
                "rejection_reasons": "|".join(rejection_reasons) if rejection_reasons else None,
                "accepted": len(rejection_reasons) == 0
            })
            
            if associated:
                association_log.append({
                    "frame": frame_idx,
                    "face_id": face_idx,
                    "track_id": best_track_id,
                    "iou": best_iou,
                    "face_area_pct": area_frac * 100,
                    "face_conf": face.score
                })
            
            # Draw on frame if saving
            if args.save_frames:
                color = (0, 255, 0) if not rejection_reasons else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{face.score:.2f}"
                if rejection_reasons:
                    label += f" [{rejection_reasons[0]}]"
                if associated:
                    label += f" T{best_track_id}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw person tracks
        if args.save_frames:
            for obs in observations:
                x1, y1, x2, y2 = obs.bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"T{obs.track_id}", (int(x1), int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save frame
            if frame_idx % args.every_nth == 0 or frame_idx in [frames_to_analyze[0], frames_to_analyze[-1]]:
                output_path = args.output / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_path), frame)
        
        # Progress
        if frames_analyzed % 10 == 0:
            LOGGER.info("Analyzed %d/%d frames...", frames_analyzed, len(frames_to_analyze))
    
    cap.release()
    
    # Generate summary report
    LOGGER.info("")
    LOGGER.info("="*60)
    LOGGER.info("DETECTION SUMMARY")
    LOGGER.info("="*60)
    
    total_faces = rejection_stats["accepted"] + rejection_stats["total_rejected"]
    
    for reason, count in rejection_stats.most_common():
        pct = (count / total_faces * 100) if total_faces > 0 else 0
        LOGGER.info("%-30s %6d  (%5.1f%%)", reason, count, pct)
    
    LOGGER.info("")
    LOGGER.info("Total faces detected:      %d", total_faces)
    if total_faces > 0:
        LOGGER.info("Acceptance rate:           %.1f%%", 
                   rejection_stats["accepted"] / total_faces * 100)
    LOGGER.info("Frames analyzed:           %d", frames_analyzed)
    LOGGER.info("Avg faces per frame:       %.1f", total_faces / frames_analyzed if frames_analyzed else 0)
    
    # Save detailed logs
    if detection_log:
        df = pd.DataFrame(detection_log)
        csv_path = args.output / "detection_log.csv"
        df.to_csv(csv_path, index=False)
        LOGGER.info("")
        LOGGER.info("Detailed detection log: %s", csv_path)
        
        # Summary by rejection reason
        if "rejection_reasons" in df.columns:
            df_rejected = df[df["rejection_reasons"].notna()]
            if len(df_rejected) > 0:
                summary_path = args.output / "rejection_summary.csv"
                reason_counts = df_rejected["rejection_reasons"].value_counts()
                reason_counts.to_csv(summary_path, header=["count"])
                LOGGER.info("Rejection summary:      %s", summary_path)
    
    if association_log:
        df_assoc = pd.DataFrame(association_log)
        assoc_path = args.output / "association_log.csv"
        df_assoc.to_csv(assoc_path, index=False)
        LOGGER.info("Association log:        %s", assoc_path)
    
    # Generate recommendations
    LOGGER.info("")
    LOGGER.info("="*60)
    LOGGER.info("RECOMMENDATIONS")
    LOGGER.info("="*60)
    
    recommendations = []
    
    if rejection_stats["area_too_small"] > total_faces * 0.2:
        current_frac = config.get("min_area_frac", 0.005)
        recommended_frac = current_frac * 0.5
        recommendations.append(
            f"• Many faces rejected for being too small ({rejection_stats['area_too_small']} faces)\n"
            f"  Consider lowering min_area_frac: {current_frac:.4f} → {recommended_frac:.4f}"
        )
    
    if rejection_stats["low_confidence"] > total_faces * 0.15:
        current_thresh = config.get("face_conf_th", 0.45)
        recommended_thresh = max(0.25, current_thresh - 0.10)
        recommendations.append(
            f"• Many faces rejected for low confidence ({rejection_stats['low_confidence']} faces)\n"
            f"  Consider lowering face_conf_th: {current_thresh:.2f} → {recommended_thresh:.2f}"
        )
    
    if rejection_stats["no_track_association"] > total_faces * 0.25:
        current_iou = config.get("face_in_track_iou", 0.25)
        recommended_iou = max(0.05, current_iou * 0.5)
        current_dilate = config.get("dilate_track_px", 0.07)
        recommended_dilate = min(0.30, current_dilate * 1.5)
        recommendations.append(
            f"• Many faces not associated with tracks ({rejection_stats['no_track_association']} faces)\n"
            f"  Consider:\n"
            f"    - Lowering face_in_track_iou: {current_iou:.2f} → {recommended_iou:.2f}\n"
            f"    - Increasing dilate_track_px: {current_dilate:.2f} → {recommended_dilate:.2f}"
        )
    
    if rejection_stats["accepted"] / total_faces > 0.8:
        recommendations.append("• Detection configuration looks good! Acceptance rate > 80%")
    
    if recommendations:
        for rec in recommendations:
            LOGGER.info(rec)
    else:
        LOGGER.info("• No specific recommendations at this time")
    
    LOGGER.info("="*60)


if __name__ == "__main__":
    main()
