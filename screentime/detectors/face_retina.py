"""RetinaFace detection and alignment utilities."""

from __future__ import annotations

import logging
import os
import platform
from typing import List, Optional, Tuple

import cv2
import numpy as np

from screentime.types import BBox, Detection

LOGGER = logging.getLogger("screentime.detectors.face")


def _default_providers() -> Tuple[str, ...]:
    """Choose default ONNX providers for RetinaFace."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


class RetinaFaceDetector:
    """Wrapper around InsightFace RetinaFace detector with alignment utilities."""

    def __init__(
        self,
        providers: Optional[Tuple[str, ...]] = None,
        det_size: Tuple[int, int] = (960, 960),
        det_thresh: float = 0.45,
    ) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "insightface is required for RetinaFaceDetector. "
                "Install it via `pip install insightface`."
            ) from exc

        self.det_size = det_size
        self.det_thresh = det_thresh
        provider_list: Tuple[str, ...]
        if providers is None:
            provider_list = _default_providers()
        else:
            provider_list = tuple(providers)
        self.providers = provider_list
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=list(provider_list))
        ctx_id = 0  # auto GPU/CoreML if available
        self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        backend = None
        try:
            detection_model = self.app.models.get("detection")
            if detection_model is not None and hasattr(detection_model, "session"):
                backend = detection_model.session.get_providers()[0]
        except Exception:  # pragma: no cover - optional logging
            backend = None
        LOGGER.info(
            "Loaded RetinaFace detector det_size=%s det_thresh=%.2f providers=%s backend=%s",
            det_size,
            det_thresh,
            provider_list,
            backend,
        )

    def detect(self, image: np.ndarray, frame_idx: int) -> List[Detection]:
        """Run RetinaFace on an image and return detections."""
        faces = self.app.get(image)
        detections: List[Detection] = []
        for face in faces:
            score = float(face.det_score)
            if score < self.det_thresh:
                continue
            bbox = tuple(float(v) for v in face.bbox)  # type: ignore[assignment]
            landmarks = np.asarray(face.kps, dtype=np.float32) if face.kps is not None else None
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox=bbox,  # type: ignore[arg-type]
                    score=score,
                    class_id=0,
                    landmarks=landmarks,
                )
            )
        return detections

    @staticmethod
    def align_to_112(image: np.ndarray, landmarks: Optional[np.ndarray], bbox: BBox) -> np.ndarray:
        """Align face to 112x112 using landmarks if available, else simple crop+resize."""
        target_size = (112, 112)
        if landmarks is None or landmarks.shape != (5, 2):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                crop = image
            return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

        src = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        dst = landmarks.astype(np.float32)
        trans = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
        if trans is None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                crop = image
            return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

        aligned = cv2.warpAffine(image, trans, target_size, borderValue=0.0)
        return aligned
