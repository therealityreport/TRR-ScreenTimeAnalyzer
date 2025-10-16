#!/usr/bin/env python3
"""Extract a small clip and run harvest to debug rejection issues."""

import argparse
import subprocess
from pathlib import Path

def extract_clip(video_path: Path, start_sec: float, duration_sec: float, output_path: Path):
    """Extract a clip using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', str(video_path),
        '-t', str(duration_sec),
        '-c', 'copy',
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✅ Extracted clip: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract clip and debug harvest")
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/debug_harvest"))
    args = parser.parse_args()
    
    # Extract clip
    clip_path = args.output_dir / f"clip_{args.start:.0f}s.mp4"
    extract_clip(args.video, args.start, args.duration, clip_path)
    
    # Run harvest with debug enabled
    print(f"\n🔍 Running harvest with debug_rejections=true...")
    print(f"   Look for rejection reasons in the logs\n")
    
    harvest_cmd = [
        'python', 'scripts/harvest_faces.py',
        str(clip_path),
        '--person-weights', 'models/weights/yolov8n.pt',
        '--output-dir', str(args.output_dir / 'harvest'),
        '--onnx-providers', 'CPUExecutionProvider'
    ]
    
    subprocess.run(harvest_cmd)
    
    print(f"\n✅ Check results in: {args.output_dir / 'harvest'}")
    print(f"   Look at the logs above for 'REJECTED' messages")

if __name__ == "__main__":
    main()
