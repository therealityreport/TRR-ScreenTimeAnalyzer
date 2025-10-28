"""RetinaFace detection and alignment utilities."""

from __future__ import annotations

import logging
import os
import platform
from typing import List, Optional, Tuple
from typing import TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - optional dependency for test environment
    import cv2  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised in tests
    cv2 = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type hints only
    import cv2 as cv2_module  # noqa: F401

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
        # CoreML-backed inference yields stable five-point landmarks. On CPU-only
        # runs the landmarks can wobble enough to hurt recognition, so we fall back
        # to simple bbox crops when no accelerated provider is available.
        self.force_bbox_alignment = provider_list == ("CPUExecutionProvider",)
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
        crop = _crop_to_bbox(image, bbox)
        if landmarks is None or landmarks.shape != (5, 2):
            return _resize_image(crop, target_size)

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
        trans = None
        if cv2 is not None and hasattr(cv2, "estimateAffinePartial2D"):
            trans = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
        if trans is None:
            return _resize_image(crop, target_size)

        if cv2 is not None and hasattr(cv2, "warpAffine"):
            aligned = cv2.warpAffine(image, trans, target_size, borderValue=0.0)
        else:
            aligned = _warp_affine(image, trans, target_size)
        return aligned


def _crop_to_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return image.copy()
    return crop


def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    width, height = [max(1, int(v)) for v in target_size]
    if image.ndim == 2:
        image = image[:, :, None]
    src_h, src_w = image.shape[:2]
    if src_h == height and src_w == width:
        return image.copy()
    if cv2 is not None and hasattr(cv2, "resize"):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    y_idx = np.clip(np.round(np.linspace(0, src_h - 1, height)).astype(int), 0, src_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, src_w - 1, width)).astype(int), 0, src_w - 1)
    resized = image[np.ix_(y_idx, x_idx)]
    if resized.ndim == 2:
        return resized
    return resized.reshape(height, width, image.shape[2])


def _warp_affine(image: np.ndarray, matrix: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    width, height = [max(1, int(v)) for v in target_size]
    if image.ndim == 2:
        image = image[:, :, None]
    src_h, src_w = image.shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    ones = np.ones_like(grid_x, dtype=np.float32)
    coords = np.stack([grid_x.astype(np.float32), grid_y.astype(np.float32), ones], axis=-1)
    inv = np.linalg.pinv(np.vstack([matrix, [0.0, 0.0, 1.0]])).astype(np.float32)
    mapped = coords @ inv.T
    mapped_x = np.clip(mapped[..., 0], 0, src_w - 1)
    mapped_y = np.clip(mapped[..., 1], 0, src_h - 1)
    x_idx = np.round(mapped_x).astype(int)
    y_idx = np.round(mapped_y).astype(int)
    warped = image[y_idx, x_idx]
    if warped.ndim == 2:
        return warped
    return warped.reshape(height, width, image.shape[2])
