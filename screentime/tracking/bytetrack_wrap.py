"""Wrapper for ByteTrack multi-object tracker integration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from screentime.types import BBox, Detection, TrackState

LOGGER = logging.getLogger("screentime.tracking.bytetrack")


@dataclass
class TrackObservation:
    track_id: int
    bbox: BBox
    score: float


class ByteTrackWrapper:
    """Adapter around the Ultralytics BYTETracker implementation."""

    def __init__(
        self,
        track_buffer: int = 30,
        match_thresh: float = 0.7,
        conf_thres: float = 0.1,
        fuse_score: bool = True,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
    ) -> None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker

            class _Args:
                def __init__(self) -> None:
                    self.track_thresh = conf_thres
                    self.track_buffer = track_buffer
                    self.match_thresh = match_thresh
                    self.fuse_score = fuse_score
                    self.mot20 = False
                    self.track_high_thresh = track_high_thresh
                    self.track_low_thresh = track_low_thresh
                    self.new_track_thresh = new_track_thresh

            self.tracker = BYTETracker(_Args(), frame_rate=30)
            self._fallback = False
            self._track_buffer = track_buffer
        except Exception as exc:  # pragma: no cover - import guard
            LOGGER.warning("Falling back to simple IOU tracker: %s", exc)
            self.tracker = _SimpleTracker(track_buffer=track_buffer)
            self._fallback = True
            self._track_buffer = track_buffer
        LOGGER.info(
            "Initialised ByteTrackWrapper track_buffer=%s match_thresh=%.2f conf_thres=%.2f",
            track_buffer,
            match_thresh,
            conf_thres,
        )
        self._class_names = {0: "person"}

    def set_frame_rate(self, fps: float) -> None:
        if fps <= 0:
            return
        if hasattr(self.tracker, "frame_rate"):
            self.tracker.frame_rate = fps

    def _detections_to_numpy(self, detections: Sequence[Detection]) -> np.ndarray:
        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            rows.append([x1, y1, x2, y2, det.score, det.class_id])
        if not rows:
            return np.zeros((0, 6), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def update(self, detections: Sequence[Detection], image_shape: Sequence[int]) -> List[TrackObservation]:
        """Update tracker with current detections."""
        if len(image_shape) < 2:
            raise ValueError("image_shape must be (height, width[, channels])")
        height, width = image_shape[:2]
        dets = self._detections_to_numpy(detections)
        if self._fallback:
            online_targets = self.tracker.update(dets)
            tracker_mode = "fallback"
        else:
            try:
                results = _build_results_from_dets(dets, (height, width))
                online_targets = self.tracker.update(results, img=None, feats=None)
                tracker_mode = "byte_numpy" if isinstance(online_targets, np.ndarray) else "byte_objects"
            except Exception as exc:  # pragma: no cover - runtime guard
                LOGGER.warning("ByteTrack update failed (%s); switching to IOU tracker fallback", exc)
                self.tracker = _SimpleTracker(track_buffer=self._track_buffer)
                self._fallback = True
                online_targets = self.tracker.update(dets)
                tracker_mode = "fallback"

        observations: List[TrackObservation] = []
        for track in list(np.asarray(online_targets)) if tracker_mode == "byte_numpy" else online_targets:
            if tracker_mode == "fallback":
                x1, y1, x2, y2, score, tid = track
                bbox: BBox = (float(x1), float(y1), float(x2), float(y2))
                track_id = int(tid)
                track_score = float(score)
            elif tracker_mode == "byte_numpy":
                x1, y1, x2, y2, tid, score = track[:6]
                bbox = (float(x1), float(y1), float(x2), float(y2))
                track_id = int(tid)
                track_score = float(score)
            else:
                tlwh = track.tlwh
                x1, y1, w, h = tlwh
                bbox = (x1, y1, x1 + w, y1 + h)
                track_id = track.track_id
                track_score = float(track.score)
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    bbox=bbox,
                    score=track_score,
                )
            )
        return observations


class TrackAccumulator:
    """Maintains metadata for active and finished tracks."""

    def __init__(self) -> None:
        self.active: Dict[int, TrackState] = {}
        self.finished: List[TrackState] = []

    def update(
        self,
        frame_idx: int,
        timestamp_ms: float,
        observations: Sequence[TrackObservation],
    ) -> None:
        seen_ids = set()
        for obs in observations:
            seen_ids.add(obs.track_id)
            state = self.active.get(obs.track_id)
            if state is None:
                state = TrackState(track_id=obs.track_id)
                self.active[obs.track_id] = state
            state.add_observation(frame_idx, obs.bbox, obs.score, timestamp_ms)

        lost_ids = [tid for tid in list(self.active) if tid not in seen_ids]
        for tid in lost_ids:
            state = self.active.pop(tid)
            state.active = False
            self.finished.append(state)

    def flush(self) -> List[TrackState]:
        """Mark remaining active tracks as finished and return all."""
        for tid, state in list(self.active.items()):
            state.active = False
            self.finished.append(state)
            self.active.pop(tid, None)
        finished_tracks = list(self.finished)
        self.finished.clear()
        return finished_tracks


class _SimpleTracker:
    """Very small IOU-based tracker used when ByteTrack is unavailable."""

    def __init__(self, track_buffer: int = 30) -> None:
        self.next_id = 1
        self.track_buffer = track_buffer
        self.tracks: Dict[int, Dict] = {}

    def update(self, detections: np.ndarray) -> List[Tuple[float, float, float, float, float, int]]:
        outputs: List[Tuple[float, float, float, float, float, int]] = []
        assigned = set()
        for det in detections:
            x1, y1, x2, y2, score, _ = det
            matched_id = None
            for tid, track in list(self.tracks.items()):
                iou_score = _iou((x1, y1, x2, y2), track["bbox"])
                if iou_score > 0.3:
                    matched_id = tid
                    break
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            self.tracks[matched_id] = {"bbox": (x1, y1, x2, y2), "score": score, "age": 0}
            assigned.add(matched_id)
            outputs.append((x1, y1, x2, y2, score, matched_id))

        for tid in list(self.tracks.keys()):
            if tid not in assigned:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.track_buffer:
                    self.tracks.pop(tid)

        return outputs


def _iou(box_a: BBox, box_b: BBox) -> float:
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
    return inter_area / union if union > 0 else 0.0


def _build_results_from_dets(dets: np.ndarray, image_shape: Tuple[int, int]) -> "Results":
    """Create a minimal Ultralytics Results object for ByteTrack."""
    from types import SimpleNamespace

    height, width = image_shape
    if dets.size == 0:
        conf = np.zeros((0,), dtype=np.float32)
        xyxy = np.zeros((0, 4), dtype=np.float32)
        cls = np.zeros((0,), dtype=np.float32)
    else:
        conf = dets[:, 4].astype(np.float32)
        xyxy = dets[:, :4].astype(np.float32)
        cls = dets[:, 5].astype(np.float32)

    class _ResultsAdapter:
        def __init__(self, conf: np.ndarray, xyxy: np.ndarray, cls: np.ndarray) -> None:
            self.conf = conf
            self.xyxy = xyxy
            self.cls = cls

        def __getitem__(self, item) -> "_ResultsAdapter":
            return _ResultsAdapter(self.conf[item], self.xyxy[item], self.cls[item])

        def __len__(self) -> int:
            return len(self.conf)

        @property
        def shape(self) -> Tuple[int, int]:
            return self.xyxy.shape

        @property
        def xywh(self) -> np.ndarray:
            if self.xyxy.size == 0:
                return np.zeros((0, 4), dtype=np.float32)
            x1 = self.xyxy[:, 0]
            y1 = self.xyxy[:, 1]
            x2 = self.xyxy[:, 2]
            y2 = self.xyxy[:, 3]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            return np.stack([cx, cy, w, h], axis=-1).astype(np.float32)

    return _ResultsAdapter(conf, xyxy, cls)
