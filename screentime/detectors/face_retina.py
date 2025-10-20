"""RetinaFace detection and alignment utilities."""

from __future__ import annotations

import logging
import platform
from typing import List, Optional, Tuple

import cv2
import numpy as np

from screentime.types import BBox, Detection

try:  # pragma: no cover - optional dependency import guard
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

LOGGER = logging.getLogger("screentime.detectors.face")


def _default_providers() -> Tuple[str, ...]:
    """Choose default ONNX providers for RetinaFace."""
    if ort is None:
        return ("CPUExecutionProvider",)
    available = {provider for provider in ort.get_available_providers()}
    providers: List[str] = []
    if "CoreMLExecutionProvider" in available and platform.machine().lower() == "arm64":
        providers.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return tuple(providers) if providers else ("CPUExecutionProvider",)


class RetinaFaceDetector:
    """Wrapper around InsightFace RetinaFace detector with alignment utilities."""

    def __init__(
        self,
        providers: Optional[Tuple[str, ...]] = None,
        det_size: Tuple[int, int] = (960, 960),
        det_thresh: float = 0.45,
        threads: int = 1,
        session_options=None,
        user_det_size_override: bool = False,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "insightface is required for RetinaFaceDetector. "
                "Install it via `pip install insightface`."
            ) from exc

        provider_list: Tuple[str, ...]
        if providers is None:
            provider_list = _default_providers()
        else:
            provider_list = tuple(providers)
            if "CPUExecutionProvider" not in provider_list:
                provider_list = tuple(list(provider_list) + ["CPUExecutionProvider"])

        session_opts = session_options
        if session_opts is None and ort is not None:
            try:
                session_opts = ort.SessionOptions()
                session_opts.intra_op_num_threads = max(1, int(threads))
                session_opts.inter_op_num_threads = 1
            except Exception:  # pragma: no cover - optional
                session_opts = None

        det_size_tuple = tuple(int(v) for v in det_size)
        if (
            not user_det_size_override
            and "CoreMLExecutionProvider" in provider_list
            and det_size_tuple == (960, 960)
        ):
            det_size_tuple = (640, 640)
            LOGGER.info("RetinaFace defaulted det_size to %s for CoreML provider", det_size_tuple)

        self.det_size = det_size_tuple
        self.det_thresh = det_thresh
        self.providers = provider_list
        # CoreML-backed inference yields stable five-point landmarks. On CPU-only
        # runs the landmarks can wobble enough to hurt recognition, so we fall back
        # to simple bbox crops when no accelerated provider is available.
        self.force_bbox_alignment = not any(p != "CPUExecutionProvider" for p in self.providers)
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection"],
            providers=list(provider_list),
            sess_options=session_opts,
        )
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
            self.det_size,
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
    def align_to_112(
        image: np.ndarray,
        landmarks: Optional[np.ndarray],
        bbox: BBox,
        force_bbox: bool = False,
    ) -> np.ndarray:
        """Align face to 112x112 using landmarks if available, else simple crop+resize."""
        target_size = (112, 112)
        if (
            force_bbox
            or landmarks is None
            or not isinstance(landmarks, np.ndarray)
            or landmarks.shape != (5, 2)
        ):
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
        # Degenerate transforms can yield mostly-empty output; fall back if we hit that.
        if aligned.size == 0 or float(np.count_nonzero(aligned)) / float(aligned.size) < 0.05:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                crop = image
            return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        return aligned
