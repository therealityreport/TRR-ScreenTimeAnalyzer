"""ArcFace embedding utilities."""

from __future__ import annotations

import logging
<<<<<<< HEAD
=======
import os
>>>>>>> origin/feat/identity-guard
import platform
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from screentime.types import l2_normalize

<<<<<<< HEAD
try:  # pragma: no cover - optional dependency
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

=======
>>>>>>> origin/feat/identity-guard
LOGGER = logging.getLogger("screentime.recognition.embed")


def _default_providers() -> Tuple[str, ...]:
    """Choose default ONNX providers based on platform."""
<<<<<<< HEAD
    if ort is None:
        return ("CPUExecutionProvider",)
    available = {provider for provider in ort.get_available_providers()}
    providers: list[str] = []
    if "CoreMLExecutionProvider" in available and platform.machine().lower() == "arm64":
        providers.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return tuple(providers) if providers else ("CPUExecutionProvider",)
=======
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)
>>>>>>> origin/feat/identity-guard


class ArcFaceEmbedder:
    """Loads an ArcFace ONNX model via InsightFace for embedding extraction."""

<<<<<<< HEAD
    def __init__(
        self,
        model_path: Optional[str] = None,
        providers: Optional[Sequence[str]] = None,
        threads: int = 1,
        session_options=None,
    ) -> None:
=======
    def __init__(self, model_path: Optional[str] = None, providers: Optional[Sequence[str]] = None) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", "2")
>>>>>>> origin/feat/identity-guard
        try:
            from insightface.model_zoo import get_model
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "insightface is required for ArcFaceEmbedder. "
                "Install it via `pip install insightface`."
            ) from exc

        if model_path:
            resolved = str(Path(model_path).expanduser())
        else:
            resolved = "arcface_r100_v1"
        provider_list: Tuple[str, ...]
        if providers is None:
            provider_list = _default_providers()
        else:
            provider_list = tuple(providers)
<<<<<<< HEAD
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

        LOGGER.info("Loading ArcFace model %s providers=%s", resolved, provider_list)
        model = get_model(resolved, download=True, providers=list(provider_list), sess_options=session_opts)
=======
        LOGGER.info("Loading ArcFace model %s providers=%s", resolved, provider_list)
        model = get_model(resolved, download=True, providers=list(provider_list))
>>>>>>> origin/feat/identity-guard
        if model is None:
            LOGGER.info("Falling back to FaceAnalysis recognition model")
            from insightface.app import FaceAnalysis

<<<<<<< HEAD
            analysis = FaceAnalysis(
                name="buffalo_l",
                providers=list(provider_list),
                sess_options=session_opts,
            )
=======
            analysis = FaceAnalysis(name="buffalo_l", providers=list(provider_list))
>>>>>>> origin/feat/identity-guard
            analysis.prepare(ctx_id=0)
            model = analysis.models.get("recognition")
            if model is None:
                raise RuntimeError("Unable to load ArcFace recognition model via insightface FaceAnalysis")
        if hasattr(model, "prepare"):
            model.prepare(ctx_id=0)
        self.model = model
        self.providers = provider_list
        self.backend = None
        try:
            session = getattr(model, "session", None)
            if session is not None:
                self.backend = session.get_providers()[0]
        except Exception:  # pragma: no cover - provider introspection best-effort
            self.backend = None

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Compute L2-normalized embedding for an aligned 112x112 face image."""
        feat = self.model.get_feat(aligned_face)
        embedding = l2_normalize(np.asarray(feat, dtype=np.float32).reshape(-1))
        return embedding
