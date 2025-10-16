#!/usr/bin/env python3
"""Diagnose why specific faces aren't being harvested."""

import cv2
from pathlib import Path

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper
from screentime.io_utils import load_yaml, setup_logging

import logging
LOGGER = logging.getLogger(__name__)


def diagnose_frame(video_path: Path, frame_idx: int, pipeline_cfg: dict, tracker_cfg: dict):
    """Diagnose what happens at a specific frame."""
    
    # Load video
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return
    
    frame_h, frame_w = frame.shape[:2]
    
    # Initialize detectors
    person_det = YOLOPersonDetector(
        weights="models/weights/yolov8n.pt",
        conf_thres=pipeline_cfg.get("person_conf_th", 0.10)
    )
    face_det = RetinaFaceDetector(
        det_size=(640, 640),
        det_thresh=pipeline_cfg.get("face_conf_th", 0.30),
        providers=("CPUExecutionProvider",)
    )
    
    print(f"\n{'='*80}")
    print(f"FRAME {frame_idx} DIAGNOSIS")
    print(f"{'='*80}")
    
    # Detect persons
    person_dets = person_det.detect(frame, frame_idx)
    print(f"\n👤 PERSON DETECTIONS: {len(person_dets)}")
    for i, det in enumerate(person_dets):
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1
        print(f"   Person {i+1}: bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) " +
              f"size={int(w)}x{int(h)} conf={det.score:.3f}")
        
        # Show dilated box
        dilate = pipeline_cfg.get("dilate_track_px", 0.15)
        dw = w * dilate
        dh = h * dilate
        dx1 = max(0, x1 - dw)
        dy1 = max(0, y1 - dh)
        dx2 = min(frame_w, x2 + dw)
        dy2 = min(frame_h, y2 + dh)
        print(f"      Dilated (dilate={dilate}): ({int(dx1)}, {int(dy1)}, {int(dx2)}, {int(dy2)})")
    
    # Detect faces
    face_dets = face_det.detect(frame, frame_idx)
    print(f"\n😊 FACE DETECTIONS: {len(face_dets)}")
    for i, det in enumerate(face_dets):
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1
        area_frac = (w * h) / (frame_w * frame_h)
        print(f"   Face {i+1}: bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) " +
              f"size={int(w)}x{int(h)} conf={det.score:.3f} area_frac={area_frac:.5f}")
        
        # Check IoU with each person box
        if person_dets:
            print(f"      IoU with person boxes:")
            for j, pdet in enumerate(person_dets):
                px1, py1, px2, py2 = pdet.bbox
                dilate = pipeline_cfg.get("dilate_track_px", 0.15)
                pw, ph = px2 - px1, py2 - py1
                dw, dh = pw * dilate, ph * dilate
                dx1 = max(0, px1 - dw)
                dy1 = max(0, py1 - dh)
                dx2 = min(frame_w, px2 + dw)
                dy2 = min(frame_h, py2 + dh)
                
                # Calculate IoU
                ix1 = max(x1, dx1)
                iy1 = max(y1, dy1)
                ix2 = min(x2, dx2)
                iy2 = min(y2, dy2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    face_area = (x2 - x1) * (y2 - y1)
                    person_area = (dx2 - dx1) * (dy2 - dy1)
                    union = face_area + person_area - intersection
                    iou = intersection / union if union > 0 else 0
                else:
                    iou = 0
                
                threshold = pipeline_cfg.get("face_in_track_iou", 0.10)
                match = "✅ MATCH" if iou >= threshold else "❌ NO MATCH"
                print(f"        Person {j+1}: IoU={iou:.4f} (threshold={threshold}) {match}")
        else:
            print(f"      ⚠️  NO PERSON BOXES - face cannot be associated with any track!")
    
    print(f"\n{'='*80}\n")


def main():
    setup_logging()
    
    video_path = Path("data/RHOBH-TEST.mp4")
    pipeline_cfg = load_yaml(Path("configs/pipeline.yaml"))
    tracker_cfg = load_yaml(Path("configs/bytetrack.yaml"))
    
    # Diagnose Eileen's frames
    print("\n🔍 Diagnosing Eileen's appearances...\n")
    
    diagnose_frame(video_path, 215, pipeline_cfg, tracker_cfg)   # 0:09
    diagnose_frame(video_path, 1751, pipeline_cfg, tracker_cfg)  # 1:13


if __name__ == "__main__":
    main()
