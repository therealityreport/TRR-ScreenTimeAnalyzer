"""YOLO-based person detector wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from screentime.types import BBox, Detection

LOGGER = logging.getLogger("screentime.detectors.person")


class YOLOPersonDetector:
    """Thin wrapper combining Ultralytics YOLO inference with post-processing."""

    def __init__(
        self,
        weights: str,
        device: Optional[str] = None,
        conf_thres: float = 0.2,
        iou_thres: float = 0.5,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "ultralytics is required for YOLOPersonDetector. "
                "Install it via `pip install ultralytics`."
            ) from exc

        self.model = YOLO(weights)
        resolved_device = device
        if resolved_device is None:
            try:
                import torch  # type: ignore

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    resolved_device = "mps"
                elif torch.cuda.is_available():
                    resolved_device = "cuda"
            except Exception:  # pragma: no cover - optional dependency
                resolved_device = None
        if resolved_device is not None:
            try:
                self.model.to(resolved_device)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    "YOLO person detector could not use device=%s (%s); falling back to auto.",
                    resolved_device,
                    exc,
                )
                resolved_device = None
        self.device = resolved_device or "auto"
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        LOGGER.info(
            "Loaded YOLO person detector weights=%s device=%s conf=%.2f",
            weights,
            self.device,
            conf_thres,
        )

    def detect(self, image: np.ndarray, frame_idx: int) -> List[Detection]:
        """Run inference on a single frame."""
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls.item()) if box.cls is not None else -1
                if cls != 0:  # person class
                    continue
                score = float(box.conf.item()) if box.conf is not None else 0.0
                xyxy = box.xyxy.cpu().numpy().flatten()
                bbox: BBox = tuple(float(x) for x in xyxy)  # type: ignore
                detections.append(
                    Detection(frame_idx=frame_idx, bbox=bbox, score=score, class_id=cls)
                )
        return detections

    def detect_batch(self, frames: Iterable[np.ndarray], start_frame: int) -> List[List[Detection]]:
        """Run inference on an iterable of frames."""
        detections: List[List[Detection]] = []
        for offset, frame in enumerate(frames):
            detections.append(self.detect(frame, start_frame + offset))
        return detections
