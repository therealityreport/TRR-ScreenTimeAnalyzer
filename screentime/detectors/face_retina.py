"""RetinaFace detection and alignment utilities."""

from __future__ import annotations

import logging
import os
import platform
from typing import List, Optional, Sequence, Tuple

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


def _has_coreml(providers: Sequence[str]) -> bool:
    return any(provider.lower().startswith("coreml") for provider in providers)


def _dummy_shape_from_input(shape: Sequence[int]) -> List[int]:
    dims: List[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            dims.append(1)
    if not dims:
        dims = [1]
    return dims


class RetinaFaceDetector:
    """Wrapper around InsightFace RetinaFace detector with alignment utilities."""

    def __init__(
        self,
        providers: Optional[Sequence[str]] = None,
        det_size: Tuple[int, int] = (960, 960),
        det_thresh: float = 0.45,
        threads: int = 1,
        session_options: Optional[object] = None,
        user_det_size_override: bool = False,
    ) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, int(threads))))
        os.environ.setdefault("MKL_NUM_THREADS", str(max(1, int(threads))))
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(max(1, int(threads))))
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "insightface is required for RetinaFaceDetector. "
                "Install it via `pip install insightface`."
            ) from exc

        provider_list: Tuple[str, ...]
        if providers is None or not tuple(providers):
            provider_list = _default_providers()
        else:
            provider_list = tuple(providers)

        self.det_thresh = float(det_thresh)
        self.providers: Tuple[str, ...] = provider_list
        self.session_options = session_options
        self.user_det_size_override = bool(user_det_size_override)

        coreml_requested = _has_coreml(self.providers)
        if coreml_requested and not self.user_det_size_override:
            det_size = (640, 640)

        self.det_size = tuple(int(max(1, v)) for v in det_size)

        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=list(self.providers))
        ctx_id = 0  # auto GPU/CoreML if available
        self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)

        self._initialize_backend(coreml_requested)

        backend = None
        try:
            detection_model = self.app.models.get("detection")
            if detection_model is not None and hasattr(detection_model, "session"):
                providers = detection_model.session.get_providers()
                backend = providers[0] if providers else None
        except Exception:  # pragma: no cover - optional logging
            backend = None
        LOGGER.info(
            "Loaded RetinaFace detector det_size=%s det_thresh=%.2f providers=%s backend=%s",
            self.det_size,
            self.det_thresh,
            self.providers,
            backend,
        )

    def _initialize_backend(self, coreml_requested: bool) -> None:
        detection_model = self.app.models.get("detection")
        if detection_model is None:
            raise RuntimeError("RetinaFace detection model unavailable after initialization")

        model_path = getattr(detection_model, "model_file", None)
        if not model_path:
            LOGGER.warning("RetinaFace detection model missing model_file; skipping provider overrides")
            return

        providers = self.providers
        session = None

        warmup_shape = self._session_warmup_shape()
        if coreml_requested:
            try:
                session = self._build_session(model_path, providers, self.session_options, warmup_shape)
            except Exception as exc:
                fallback_size = self.det_size
                if not self.user_det_size_override:
                    fallback_size = (960, 960)
                LOGGER.warning(
                    "RetinaFace CoreML failed, falling back to CPUExecutionProvider; det_size=%dx%d",
                    fallback_size[0],
                    fallback_size[1],
                )
                if fallback_size != self.det_size:
                    self.det_size = fallback_size
                    self.app.prepare(ctx_id=0, det_size=self.det_size)
                    detection_model = self.app.models.get("detection")
                    if detection_model is None:
                        raise RuntimeError("RetinaFace detection model unavailable after CoreML fallback")
                    model_path = getattr(detection_model, "model_file", model_path)
                    warmup_shape = self._session_warmup_shape()
                providers = ("CPUExecutionProvider",)
                session = self._build_session(model_path, providers, self.session_options, warmup_shape)
            self.providers = providers
        else:
            session = self._build_session(model_path, providers, self.session_options, warmup_shape)

        detection_model.session = session
        if hasattr(detection_model, "_init_vars"):
            detection_model._init_vars()
        detection_model.prepare(ctx_id=0, det_thresh=float(self.det_thresh), input_size=self.det_size)
        self._warmup_detection(detection_model)

    def _session_warmup_shape(self) -> Tuple[int, int, int, int]:
        width, height = self.det_size
        width = max(1, int(width))
        height = max(1, int(height))
        return (1, 3, height, width)

    @staticmethod
    def _build_session(
        model_path: str,
        providers: Sequence[str],
        session_options: Optional[object],
        warmup_shape: Optional[Tuple[int, ...]] = None,
    ):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("onnxruntime is required for RetinaFaceDetector") from exc

        provider_list = list(providers)
        kwargs = {"providers": provider_list}
        if session_options is not None:
            kwargs["sess_options"] = session_options
        session = ort.InferenceSession(model_path, **kwargs)
        # Warm-up: ensure providers are valid and shapes resolve
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if inputs:
            input_meta = inputs[0]
            if warmup_shape is not None:
                dummy_shape = tuple(int(max(1, dim)) for dim in warmup_shape)
            else:
                dummy_shape = _dummy_shape_from_input(input_meta.shape)
            dummy = np.zeros(dummy_shape, dtype=np.float32)
            session.run([out.name for out in outputs], {input_meta.name: dummy})
        return session

    def _warmup_detection(self, detection_model) -> None:
        try:
            dummy_w, dummy_h = self.det_size
            dummy = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            detection_model.detect(dummy, input_size=self.det_size)
        except Exception as exc:  # pragma: no cover - best-effort warmup
            LOGGER.debug("RetinaFace warmup detect failed: %s", exc, exc_info=True)

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
