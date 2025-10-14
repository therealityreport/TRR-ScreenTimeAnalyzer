"""Common dataclasses and type aliases used across the screentime package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Bounding box order: x1, y1, x2, y2 (pixel coordinates)
BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    """Generic detection returned by detectors."""

    frame_idx: int
    bbox: BBox
    score: float
    class_id: int = 0
    landmarks: Optional[np.ndarray] = None

    def as_xywh(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1


@dataclass
class TrackState:
    """Represents the evolving state of a person track."""

    track_id: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    timestamps_ms: List[float] = field(default_factory=list)
    label: Optional[str] = None
    label_scores: Dict[str, float] = field(default_factory=dict)
    active: bool = True

    def add_observation(
        self,
        frame_idx: int,
        bbox: BBox,
        score: float,
        timestamp_ms: float,
    ) -> None:
        self.frames.append(frame_idx)
        self.bboxes.append(bbox)
        self.scores.append(score)
        self.timestamps_ms.append(timestamp_ms)

    @property
    def start_frame(self) -> Optional[int]:
        return self.frames[0] if self.frames else None

    @property
    def end_frame(self) -> Optional[int]:
        return self.frames[-1] if self.frames else None

    @property
    def duration_frames(self) -> int:
        if not self.frames:
            return 0
        return self.frames[-1] - self.frames[0] + 1

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    def record_vote(self, label: str, weight: float) -> None:
        self.label_scores[label] = self.label_scores.get(label, 0.0) + weight


@dataclass
class FaceSample:
    """Aligned face crop captured during harvest."""

    track_id: int  # harvest track id (reindexed)
    frame_idx: int
    timestamp_ms: float
    path: Path
    score: float
    bbox: BBox
    quality: float = 0.0
    sharpness: float = 0.0
    orientation: str = "unknown"
    frontalness: float = 0.0
    area_frac: float = 0.0
    person_bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    association_iou: float = 0.0
    match_mode: str = "candidate"
    match_frame_offset: int = 0
    byte_track_id: Optional[int] = None  # source ByteTrack id
    # Embedding is used transiently during harvesting for identity purity; not serialized
    embedding: Optional[np.ndarray] = None


@dataclass
class ManifestEntry:
    """Metadata entry describing a harvested track."""

    track_id: int
    total_frames: int
    avg_conf: float
    avg_area: float
    first_ts_ms: float
    last_ts_ms: float
    samples: List[FaceSample] = field(default_factory=list)
    label: Optional[str] = None
    byte_track_id: Optional[int] = None

    def to_dict(self) -> Dict:
        payload = {
            "track_id": self.track_id,
            "byte_track_id": self.byte_track_id,
            "total_frames": self.total_frames,
            "avg_conf": self.avg_conf,
            "avg_area": self.avg_area,
            "first_ts_ms": self.first_ts_ms,
            "last_ts_ms": self.last_ts_ms,
            "label": self.label,
            "samples": [
                {
                    "byte_track_id": s.byte_track_id,
                    "frame_idx": s.frame_idx,
                    "timestamp_ms": s.timestamp_ms,
                    "path": str(s.path),
                    "score": s.score,
                    "bbox": s.bbox,
                    "quality": s.quality,
                    "sharpness": s.sharpness,
                    "orientation": s.orientation,
                    "frontalness": s.frontalness,
                    "area_frac": s.area_frac,
                    "person_bbox": s.person_bbox,
                    "association_iou": s.association_iou,
                    "match_mode": s.match_mode,
                }
                for s in self.samples
            ],
        }
        return payload


@dataclass
class FacebankEntry:
    """Representation of a facebank centroid entry."""

    label: str
    embedding: np.ndarray
    sample_paths: List[Path]


def l2_normalize(vec: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """L2-normalize the input vector."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def iou(box_a: BBox, box_b: BBox) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_area(box: BBox) -> float:
    """Compute area of a bounding box."""
    x1, y1, x2, y2 = box
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def iter_batches(iterable: Iterable, batch_size: int) -> Iterable[List]:
    """Yield successive batches from an iterable."""
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
