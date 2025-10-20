"""Overlay rendering utilities for QA videos."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import cv2

LOGGER = logging.getLogger("screentime.viz.overlay")


@dataclass
class OverlayBox:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: Optional[str]
    score: Optional[float]
    quality: Optional[float] = None
    sharpness: Optional[float] = None
    box_type: str = "face"
    reason: Optional[str] = None
    association_iou: Optional[float] = None
    picked: Optional[bool] = None
    color: Optional[tuple[int, int, int]] = None


@dataclass
class FrameOverlay:
    frame_idx: int
    boxes: List[OverlayBox]


def _label_color(label: Optional[str]) -> tuple[int, int, int]:
    if not label:
        return (0, 255, 255)
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return tuple(int(x) for x in digest[:3])


def render_overlay(
    video_path: str,
    overlays: Dict[int, FrameOverlay],
    output_path: str,
    fps: Optional[float] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    highlight_track: Optional[int] = None,
    labels_only: bool = False,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = fps or input_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if end_frame is not None and frame_idx > end_frame:
            break
        overlay = overlays.get(frame_idx)
        if overlay:
            frame = _draw_boxes(
                frame,
                overlay.boxes,
                highlight_track=highlight_track,
                labels_only=labels_only,
            )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    LOGGER.info("Overlay written to %s", output_path)


def _draw_boxes(
    frame,
    boxes: Iterable[OverlayBox],
    highlight_track: Optional[int] = None,
    labels_only: bool = False,
):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.bbox)
        base_color = box.color or _label_color(box.label)
        is_highlight = highlight_track is not None and box.track_id == highlight_track
        color = (0, 0, 255) if is_highlight else base_color
        thickness = 4 if is_highlight else 2
        label = box.label or "unknown"
        if box.score is not None:
            label = f"{label} ({box.score:.2f})"
        if is_highlight:
            label = f"*{label}*"
        if not labels_only:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text_y = max(15, y1 - 10)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        metrics = []
        if box.reason:
            metrics.append(box.reason)
        if box.quality is not None:
            metrics.append(f"Q={box.quality:.2f}")
        if box.sharpness is not None:
            metrics.append(f"S={int(round(box.sharpness))}")
        if box.association_iou is not None:
            metrics.append(f"IoU={box.association_iou:.2f}")
        if metrics:
            metrics_y = min(frame.shape[0] - 5, text_y + 14)
            cv2.putText(frame, " ".join(metrics), (x1, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame
