#!/usr/bin/env python3
"""CLI for rendering overlay QA videos."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2

from screentime.io_utils import setup_logging
from screentime.viz.overlay import FrameOverlay, OverlayBox, render_overlay


LOGGER = logging.getLogger("scripts.overlay")


REASON_COLORS = {
    "picked": (0, 200, 0),
    "candidate": (0, 255, 255),
    "not_selected": (180, 180, 180),
    "rejected_low_iou": (0, 0, 255),
    "rejected_sharpness": (0, 140, 255),
    "rejected_frontalness": (211, 0, 148),
    "rejected_area": (255, 0, 0),
    "track": (255, 140, 0),
}
PERSON_COLOR = (255, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render harvest QA overlays from debug artifacts")
    parser.add_argument("video", type=Path, help="Original video path")
    parser.add_argument("--harvest-dir", type=Path, required=True, help="Directory containing harvest_debug.json")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path")
    parser.add_argument("--start-sec", type=float, default=None, help="Optional start time in seconds")
    parser.add_argument("--end-sec", type=float, default=None, help="Optional end time in seconds")
    parser.add_argument("--show-person", action="store_true", help="Draw person (ByteTrack) boxes")
    parser.add_argument("--show-faces", action="store_true", help="Draw face (RetinaFace) boxes")
    parser.add_argument("--show-iou", action="store_true", help="Annotate IoU values when available")
    parser.add_argument("--show-reason", action="store_true", help="Annotate accept/reject reason text")
    parser.add_argument("--highlight-track", type=int, default=None, help="Emphasize a specific ByteTrack id")
    parser.add_argument("--labels-only", action="store_true", help="Render labels without drawing bounding boxes")
    parser.add_argument(
        "--from-annotations",
        action="store_true",
        help="Render overlays using tracker annotations (fallback when harvest_debug.json is absent)",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=None,
        help="Optional explicit path to *-annotations.json (used with --from-annotations or on fallback)",
    )
    return parser.parse_args()


def _reason_color(reason: Optional[str]) -> tuple[int, int, int]:
    if not reason:
        return (255, 255, 255)
    return REASON_COLORS.get(reason, (255, 255, 255))


def _resolve_annotations_path(harvest_dir: Path, override: Optional[Path]) -> Optional[Path]:
    if override:
        return override

    candidates: list[Path] = []
    # 1. Direct files inside harvest_dir
    candidates.extend(sorted(harvest_dir.glob("*-annotations.json")))
    # 2. Heuristic: ../outputs/<stem>/<stem>-annotations.json
    try:
        outputs_root = harvest_dir.parent.parent / "outputs"
        outputs_dir = outputs_root / harvest_dir.name
        candidates.extend(sorted(outputs_dir.glob("*-annotations.json")))
    except Exception:
        pass

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_frame_events_from_annotations(annotations_path: Path) -> Dict[int, list[dict]]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    frame_events: Dict[int, list[dict]] = {}
    for entry in data.get("frame_annotations", []):
        frame_idx = int(entry.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        events: list[dict] = []
        for track in entry.get("tracks", []):
            bbox = track.get("bbox")
            if not bbox:
                continue
            events.append(
                {
                    "track_id": track.get("track_id"),
                    "face_bbox": bbox,
                    "person_bbox": bbox,
                    "label": track.get("label"),
                    "score": track.get("score"),
                    "reason": "track",
                }
            )
        if events:
            frame_events[frame_idx] = events
    return frame_events


def _load_frame_events(harvest_dir: Path, use_annotations: bool, annotations_path: Optional[Path]) -> Dict[int, list[dict]]:
    debug_path = harvest_dir / "harvest_debug.json"
    if not use_annotations:
        try:
            data = json.loads(debug_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            LOGGER.warning("harvest_debug.json not found at %s; falling back to annotations", debug_path)
        else:
            raw_events = data.get("frame_events", {})
            frame_events: Dict[int, list[dict]] = {}
            for frame_key, events in raw_events.items():
                frame_idx = int(frame_key)
                frame_events[frame_idx] = events
            return frame_events

    resolved = _resolve_annotations_path(harvest_dir, annotations_path)
    if not resolved:
        raise FileNotFoundError(
            f"Could not locate tracker annotations (tried override={annotations_path})"
        )
    LOGGER.info("Loading overlays from annotations %s", resolved)
    return _load_frame_events_from_annotations(resolved)


def main() -> None:
    args = parse_args()
    setup_logging()

    frame_events = _load_frame_events(args.harvest_dir, args.from_annotations, args.annotations_path)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    start_frame = int(args.start_sec * fps) if args.start_sec is not None else 0
    end_frame = int(args.end_sec * fps) if args.end_sec is not None else None
    if total_frames is not None:
        start_frame = max(0, min(start_frame, total_frames - 1))
        if end_frame is not None:
            end_frame = max(start_frame, min(end_frame, total_frames - 1))

    overlays: Dict[int, FrameOverlay] = {}
    show_faces = args.show_faces or not args.show_person  # default to faces if nothing selected
    for frame_idx, events in frame_events.items():
        if frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            continue
        boxes = []
        for event in events:
            track_id = event.get("track_id")
            reason = event.get("reason")
            if show_faces and event.get("face_bbox"):
                boxes.append(
                    OverlayBox(
                        track_id=track_id if track_id is not None else -1,
                        bbox=tuple(event["face_bbox"]),
                        label=f"Track {track_id} face" if track_id is not None else "face",
                        score=None,
                        quality=event.get("quality"),
                        sharpness=event.get("sharpness"),
                        box_type="face",
                        reason=reason.upper() if args.show_reason and reason else None,
                        association_iou=event.get("association_iou") if args.show_iou else None,
                        picked=event.get("picked"),
                        color=_reason_color(reason),
                    )
                )
            if args.show_person and event.get("person_bbox"):
                boxes.append(
                    OverlayBox(
                        track_id=track_id if track_id is not None else -1,
                        bbox=tuple(event["person_bbox"]),
                        label=f"Track {track_id} person" if track_id is not None else "person",
                        score=None,
                        box_type="person",
                        association_iou=event.get("association_iou") if args.show_iou else None,
                        color=PERSON_COLOR,
                    )
                )
        if boxes:
            overlays[frame_idx] = FrameOverlay(frame_idx=frame_idx, boxes=boxes)

    if not overlays:
        LOGGER.warning("No overlays generated for the requested window")

    render_overlay(
        str(args.video),
        overlays,
        str(args.output),
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame,
        highlight_track=args.highlight_track,
        labels_only=args.labels_only,
    )
    LOGGER.info("Overlay video written to %s", args.output)


if __name__ == "__main__":
    main()
