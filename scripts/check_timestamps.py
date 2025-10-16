#!/usr/bin/env python3
"""Quick diagnostic to check specific timestamps in a video."""

import argparse
import cv2
from pathlib import Path
from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.io_utils import setup_logging

def check_timestamp(video_path: Path, timestamp_sec: float, output_dir: Path):
    """Check what detectors see at a specific timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame at {timestamp_sec}s (frame {frame_idx})")
        return
    
    print(f"\n🔍 Checking timestamp {timestamp_sec}s (frame {frame_idx})")
    
    # Initialize detectors
    person_det = YOLOPersonDetector(weights="models/weights/yolov8n.pt", conf_thres=0.15)
    face_det = RetinaFaceDetector(det_size=(960, 960), det_thresh=0.30, providers=("CPUExecutionProvider",))
    
    # Detect persons
    person_dets = person_det.detect(frame, frame_idx)
    print(f"  👤 Found {len(person_dets)} person(s)")
    for i, det in enumerate(person_dets):
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1
        print(f"     Person {i+1}: bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) size={int(w)}x{int(h)} conf={det.score:.3f}")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"P{i+1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Detect faces
    face_dets = face_det.detect(frame, frame_idx)
    print(f"  😊 Found {len(face_dets)} face(s)")
    for i, det in enumerate(face_dets):
        x1, y1, x2, y2 = det.bbox
        w, h = x2 - x1, y2 - y1
        area_frac = (w * h) / (frame.shape[1] * frame.shape[0])
        print(f"     Face {i+1}: bbox=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}) size={int(w)}x{int(h)} conf={det.score:.3f} area_frac={area_frac:.5f}")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(frame, f"F{i+1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save annotated frame
    output_path = output_dir / f"frame_{frame_idx:06d}_t{timestamp_sec:.2f}s.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"  💾 Saved: {output_path}")
    print(f"  🖼️  Open with: open {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Check specific timestamps")
    parser.add_argument("video", type=Path)
    parser.add_argument("timestamps", type=float, nargs="+", help="Timestamps in seconds (e.g., 9.0 73.0)")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/timestamp_check"))
    args = parser.parse_args()
    
    setup_logging()
    
    for ts in args.timestamps:
        check_timestamp(args.video, ts, args.output)
    
    print(f"\n✅ Done! Check images in: {args.output}")
    print(f"   Open folder: open {args.output}")

if __name__ == "__main__":
    main()
