"""Facebank build/load utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.io_utils import ensure_dir
from screentime.types import FaceSample, l2_normalize

LOGGER = logging.getLogger("screentime.recognition.facebank")


@dataclass
class FacebankArtifacts:
    parquet_path: Path
    meta_json_path: Path
    samples_csv_path: Path


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def iter_facebank_images(facebank_dir: Path) -> Iterable[FaceSample]:
    for label_dir in sorted(p for p in facebank_dir.iterdir() if p.is_dir()):
        label = label_dir.name
        for img_path in sorted(
            p for p in label_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ):
            yield FaceSample(
                track_id=-1,
                frame_idx=-1,
                timestamp_ms=-1.0,
                path=img_path,
                score=1.0,
                bbox=(0.0, 0.0, 0.0, 0.0),
            )


def build_facebank(
    facebank_dir: Path,
    output_dir: Path,
    embedder,
    aligner: Optional[RetinaFaceDetector] = None,
) -> FacebankArtifacts:
    """Build centroid embeddings for the facebank directory."""

    ensure_dir(output_dir)
    rows: List[Dict] = []
    samples_rows: List[Dict] = []
    label_embeddings: Dict[str, List[np.ndarray]] = {}

    for sample in iter_facebank_images(facebank_dir):
        label = sample.path.parent.name
        image = _load_image(sample.path)
        if aligner is not None:
            aligned = aligner.align_to_112(image, None, (0, 0, image.shape[1], image.shape[0]))
        else:
            aligned = cv2.resize(image, (112, 112))
        embedding = embedder.embed(aligned)
        label_embeddings.setdefault(label, []).append(embedding)
        samples_rows.append(
            {
                "label": label,
                "path": str(sample.path),
                "embedding_norm": float(np.linalg.norm(embedding)),
            }
        )

    for label, embeds in label_embeddings.items():
        if not embeds:
            continue
        stacked = np.stack(embeds, axis=0)
        centroid = l2_normalize(stacked.mean(axis=0))
        rows.append(
            {
                "label": label,
                "count": len(embeds),
                "embedding": centroid.astype(np.float32).tolist(),
            }
        )

    if not rows:
        raise RuntimeError(f"No facebank images found under {facebank_dir}")

    df = pd.DataFrame(rows)
    samples_df = pd.DataFrame(samples_rows)

    parquet_path = output_dir / "facebank.parquet"
    meta_json_path = output_dir / "facebank_meta.json"
    samples_csv_path = output_dir / "facebank_samples.csv"

    df.to_parquet(parquet_path, index=False)
    samples_df.to_csv(samples_csv_path, index=False)

    metadata = {
        "labels": df["label"].tolist(),
        "counts": df.set_index("label")["count"].to_dict(),
        "num_labels": len(df),
    }

    with meta_json_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    LOGGER.info("Facebank built: %s labels", len(df))
    return FacebankArtifacts(parquet_path, meta_json_path, samples_csv_path)


def load_facebank(parquet_path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    facebank: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        embedding = _normalize_embedding(row["embedding"])
        facebank[row["label"]] = embedding
    return facebank


def _normalize_embedding(raw) -> np.ndarray:
    """Convert parquet-loaded embedding column into a 1D float32 vector."""
    if isinstance(raw, np.ndarray):
        if raw.dtype == object or raw.ndim > 1:
            parts = [np.asarray(part, dtype=np.float32).ravel() for part in raw]
            arr = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
        else:
            arr = raw.astype(np.float32)
    elif isinstance(raw, list):
        if raw and isinstance(raw[0], (list, tuple, np.ndarray)):
            parts = [np.asarray(part, dtype=np.float32).ravel() for part in raw]
            arr = np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
        else:
            arr = np.asarray(raw, dtype=np.float32)
    else:
        arr = np.asarray(raw, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32)
