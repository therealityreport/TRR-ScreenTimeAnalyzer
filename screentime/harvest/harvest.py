"""Harvest pipeline orchestration."""

from __future__ import annotations

import csv
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.io_utils import dump_json, ensure_dir, infer_video_stem
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper, TrackAccumulator, TrackObservation
from screentime.types import FaceSample, ManifestEntry, TrackState, bbox_area, iou
from screentime.recognition.embed_arcface import ArcFaceEmbedder

LOGGER = logging.getLogger("screentime.harvest")


@dataclass
class HarvestConfig:
    stride: int = 1
    samples_min: int = 4
    samples_max: int = 12
    samples_per_track: int = 8
    min_gap_frames: int = 8
    min_area_frac: float = 0.005
    min_area_px: Optional[float] = None
    min_sharpness_laplacian: Optional[float] = None
    min_sharpness_pct: Optional[float] = 40.0
    min_frontalness: float = 0.35
    face_in_track_iou: float = 0.20
    allow_face_center: bool = False
    dilate_track_px: float = 0.15
    temporal_iou_tolerance: int = 1
    profile_asymmetry_thresh: float = 0.25
    quality_weights: Tuple[float, float, float] = (0.35, 0.45, 0.20)
    target_area_frac: float = 0.06
    debug_rejections: bool = False
    multi_face_per_track_guard: bool = True
    multi_face_tiebreak: str = "quality"
    fallback_head_pct: float = 0.4
    # Identity purity guard across frames
    identity_guard: bool = True
    identity_split: bool = True
    identity_sim_threshold: float = 0.62
    identity_min_picks: int = 2
    # Reindex harvest folders independent of ByteTrack ids
    reindex_harvest_tracks: bool = True


@dataclass
class SampleCandidate:
    sample: FaceSample
    quality: float
    sharpness: float
    frontalness: float
    area_frac: float
    orientation: str
    picked: bool = False
    reason: Optional[str] = None


@dataclass
class AssignedFace:
    face_idx: int
    mode: str
    overlap: float
    score: float
    person_bbox: Tuple[float, float, float, float]
    frame_offset: int = 0


@dataclass
class TrackSamplingState:
    candidates: List[SampleCandidate] = field(default_factory=list)
    _pool_limit: Optional[int] = None

    def _candidate_limit(self, config: HarvestConfig) -> int:
        if self._pool_limit is None:
            self._pool_limit = max(config.samples_max, config.samples_per_track * 3)
        return self._pool_limit

    def add_candidate(
        self,
        sample: FaceSample,
        quality: float,
        sharpness: float,
        frontalness: float,
        area_frac: float,
        orientation: str,
        config: HarvestConfig,
    ) -> Optional[FaceSample]:
        candidate = SampleCandidate(
            sample=sample,
            quality=quality,
            sharpness=sharpness,
            frontalness=frontalness,
            area_frac=area_frac,
            orientation=orientation,
        )
        self.candidates.append(candidate)
        self.candidates.sort(key=lambda c: (c.quality, -c.sample.frame_idx), reverse=True)
        removed: Optional[SampleCandidate] = None
        limit = self._candidate_limit(config)
        if len(self.candidates) > limit:
            removed = self.candidates.pop()
        return removed.sample if removed else None

    def export_samples(self, config: HarvestConfig) -> List[FaceSample]:
        if not self.candidates:
            return []

        for candidate in self.candidates:
            candidate.picked = False
            candidate.reason = None

        sorted_candidates = sorted(self.candidates, key=lambda c: c.quality, reverse=True)
        selected: List[SampleCandidate] = []

        sharp_thresholds: List[float] = []
        if config.min_sharpness_pct is not None and sorted_candidates:
            sharp_thresholds.append(
                float(np.percentile([cand.sharpness for cand in sorted_candidates], config.min_sharpness_pct))
            )
        if config.min_sharpness_laplacian is not None:
            sharp_thresholds.append(float(config.min_sharpness_laplacian))
        sharp_gate = max(sharp_thresholds) if sharp_thresholds else None

        eligible_candidates: List[SampleCandidate] = []
        for candidate in sorted_candidates:
            if candidate.frontalness < config.min_frontalness:
                candidate.reason = "rejected_frontalness"
                continue
            if sharp_gate is not None and candidate.sharpness < sharp_gate:
                candidate.reason = "rejected_sharpness"
                continue
            eligible_candidates.append(candidate)

        if not eligible_candidates:
            return []

        required = min(config.samples_per_track, len(eligible_candidates))
        min_gap = max(config.min_gap_frames, 0)

        for candidate in eligible_candidates:
            if len(selected) >= required:
                break
            if all(abs(candidate.sample.frame_idx - chosen.sample.frame_idx) >= min_gap for chosen in selected):
                candidate.picked = True
                candidate.reason = "picked"
                selected.append(candidate)

        if len(selected) < min(config.samples_min, required):
            for candidate in eligible_candidates:
                if candidate.picked:
                    continue
                candidate.picked = True
                candidate.reason = "picked"
                selected.append(candidate)
                if len(selected) >= min(config.samples_min, required):
                    break

        high_frontal_candidates = [cand for cand in eligible_candidates if cand.frontalness >= 0.5]
        if high_frontal_candidates:
            high_best = max(high_frontal_candidates, key=lambda c: c.quality)
            if not high_best.picked:
                if len(selected) >= config.samples_per_track:
                    worst = min(selected, key=lambda c: c.quality)
                    if worst is not high_best:
                        worst.picked = False
                        worst.reason = worst.reason or "not_selected"
                        selected.remove(worst)
                high_best.picked = True
                high_best.reason = "picked"
                selected.append(high_best)

        for candidate in eligible_candidates:
            if not candidate.picked and candidate.reason is None:
                candidate.reason = "not_selected"

        return [c.sample for c in selected]

    def iter_debug_rows(self, track_id: int, limit: int = 20):
        sorted_candidates = sorted(self.candidates, key=lambda c: c.quality, reverse=True)
        for candidate in sorted_candidates[:limit]:
            yield {
                "track_id": track_id,
                "byte_track_id": candidate.sample.byte_track_id,
                "frame": candidate.sample.frame_idx,
                "quality": candidate.quality,
                "sharpness": candidate.sharpness,
                "frontalness": candidate.frontalness,
                "area_frac": candidate.area_frac,
                "picked": candidate.picked,
                "reason": candidate.reason or ("picked" if candidate.picked else "not_selected"),
                "path": str(candidate.sample.path),
                "association_iou": candidate.sample.association_iou,
                "match_mode": candidate.sample.match_mode,
                "frame_offset": candidate.sample.match_frame_offset,
            }


class HarvestRunner:
    """High-level pipeline controller for harvesting aligned crops."""

    def __init__(
        self,
        person_detector: YOLOPersonDetector,
        face_detector: RetinaFaceDetector,
        tracker: ByteTrackWrapper,
        config: HarvestConfig,
        embedder: Optional[ArcFaceEmbedder] = None,
    ) -> None:
        self.person_detector = person_detector
        self.face_detector = face_detector
        self.tracker = tracker
        self.config = config
        # ArcFace embedder for identity consistency checks (lazy)
        self.embedder = embedder

    def run(self, video_path: Path, output_root: Path) -> Path:
        # Lazy init embedder to keep tests lightweight; fall back if unavailable
        if self.config.identity_guard and self.embedder is None:
            try:
                self.embedder = ArcFaceEmbedder()
            except Exception as exc:  # pragma: no cover - runtime guard
                LOGGER.warning("ArcFace embedder unavailable (%s); disabling identity_guard for this run", exc)
                self.embedder = None
                # disable guard for current run to avoid errors
                self.config.identity_guard = False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_area = float(width * height)
        self.tracker.set_frame_rate(fps)

        video_stem = infer_video_stem(video_path)
        harvest_dir = ensure_dir(output_root / video_stem)
        crops_dir = ensure_dir(harvest_dir)

        accumulator = TrackAccumulator()
        sampling_states: Dict[int, TrackSamplingState] = {}
        debug_records: Dict[str, List[Dict]] = defaultdict(list)
        reject_counter: Counter[str] = Counter()
        frame_events: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        candidate_event_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # Harvest track namespace and identity guard state
        next_harvest_id = 1
        byte_to_harvest: Dict[int, int] = {}
        harvest_centroid: Dict[int, np.ndarray] = {}
        harvest_count: Dict[int, int] = {}
        harvest_source_byte: Dict[int, int] = {}

        def _new_harvest_id() -> int:
            nonlocal next_harvest_id
            hid = next_harvest_id
            next_harvest_id += 1
            return hid

        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            num = float((a * b).sum())
            den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            return num / den

        frame_idx = -1
        processed_frames = 0

        LOGGER.info(
            "Starting harvest video=%s fps=%.2f frames=%s output=%s",
            video_path,
            fps,
            frame_count or "unknown",
            harvest_dir,
        )
        if frame_count:
            progress_step = max(1, frame_count // 20)  # ~5% increments
        else:
            progress_step = 500  # fallback when frame count is unknown
        next_progress_log = progress_step
        last_frames_seen = 0

        def log_progress(frames_seen: int) -> None:
            nonlocal next_progress_log
            if frames_seen < next_progress_log:
                return
            pct = (frames_seen / frame_count * 100.0) if frame_count else None
            samples_total = sum(len(state.candidates) for state in sampling_states.values())
            if pct is not None:
                LOGGER.info(
                    "Harvest progress %.1f%% (%d/%d frames seen, processed=%d, active_tracks=%d, samples=%d)",
                    pct,
                    frames_seen,
                    frame_count,
                    processed_frames,
                    len(accumulator.active),
                    samples_total,
                )
            else:
                LOGGER.info(
                    "Harvest progress: %d frames seen (processed=%d, active_tracks=%d, samples=%d)",
                    frames_seen,
                    processed_frames,
                    len(accumulator.active),
                    samples_total,
                )
            while next_progress_log <= frames_seen:
                next_progress_log += progress_step

        def log_frame_event(frame_idx: int, payload: Dict[str, Any]) -> None:
            event = dict(payload)
            event.setdefault("frame", frame_idx)
            frame_events[frame_idx].append(event)

        def record_reject(track_id: Optional[int], reason: str, frame_idx: int, extra: Optional[Dict] = None) -> None:
            reject_counter[reason] += 1
            if not self.config.debug_rejections:
                if extra is None:
                    extra = {}
                log_frame_event(frame_idx, {"track_id": track_id, "reason": reason, **extra})
                return
            payload = {"frame_idx": frame_idx, "reason": reason}
            if extra:
                payload.update(extra)
            key = str(track_id) if track_id is not None else "unassigned"
            debug_records[key].append(payload)
            log_frame_event(frame_idx, {"track_id": track_id, **payload})

        def record_candidate_event(track_id: int, frame_idx: int, payload: Dict[str, Any]) -> Dict[str, Any]:
            event = {
                "track_id": track_id,
                "reason": payload.get("reason", "candidate"),
                **{k: v for k, v in payload.items() if k != "reason"},
            }
            event.setdefault("frame", frame_idx)
            frame_events[frame_idx].append(event)
            candidate_event_lookup[(track_id, frame_idx)] = event
            return event

        def manage_sample_file(sample: FaceSample, reason: str, event: Optional[Dict[str, Any]] = None) -> None:
            if not sample.path:
                return
            path = Path(sample.path)
            try:
                if self.config.debug_rejections:
                    debug_dir = ensure_dir(path.parent / "debug")
                    target = debug_dir / path.name
                    if path != target:
                        if target.exists():
                            target.unlink()
                        path.replace(target)
                        sample.path = target
                        if event is not None:
                            event["path"] = str(target)
                else:
                    path.unlink(missing_ok=True)
                    if event is not None:
                        event["path"] = None
            except OSError as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to manage sample file %s (%s): %s", path, reason, exc)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frames_seen = frame_idx + 1
            if frame_idx % self.config.stride != 0:
                log_progress(frames_seen)
                continue

            processed_frames += 1
            timestamp_ms = (frame_idx / fps) * 1000.0

            person_dets = self.person_detector.detect(frame, frame_idx)
            observations = self.tracker.update(person_dets, frame.shape)
            accumulator.update(frame_idx, timestamp_ms, observations)
            track_lookup = {obs.track_id: obs for obs in observations}

            face_dets = self.face_detector.detect(frame, frame_idx)
            assignments: Dict[int, AssignedFace] = {}
            unmatched_faces = set(range(len(face_dets)))

            def prefer_candidate(current: AssignedFace, candidate: AssignedFace) -> bool:
                if self.config.multi_face_tiebreak == "quality":
                    if candidate.score > current.score + 1e-6:
                        return True
                    if current.score > candidate.score + 1e-6:
                        return False
                return candidate.overlap > current.overlap

            track_box_cache: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}

            def track_candidate_boxes(track_id: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
                cached = track_box_cache.get(track_id)
                if cached is not None:
                    return cached
                boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
                state = accumulator.active.get(track_id)
                tolerance = max(0, self.config.temporal_iou_tolerance)
                if state:
                    for past_frame, bbox in zip(reversed(state.frames), reversed(state.bboxes)):
                        delta = frame_idx - past_frame
                        if delta < 0:
                            continue
                        if delta > tolerance:
                            break
                        boxes.append((delta, bbox))
                else:
                    obs = track_lookup.get(track_id)
                    if obs:
                        boxes.append((0, obs.bbox))
                track_box_cache[track_id] = boxes
                return boxes

            match_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

            if face_dets and track_lookup:
                track_ids = list(track_lookup.keys())
                num_faces = len(face_dets)
                num_tracks = len(track_ids)
                cost = np.full((num_faces, num_tracks), fill_value=1e6, dtype=np.float32)

                for i, face in enumerate(face_dets):
                    for j, track_id in enumerate(track_ids):
                        best_overlap = 0.0
                        best_bbox: Optional[Tuple[float, float, float, float]] = None
                        best_offset = 0
                        for frame_offset, cand_bbox in track_candidate_boxes(track_id):
                            dilated = self._dilate_bbox(
                                cand_bbox,
                                self.config.dilate_track_px,
                                width,
                                height,
                            )
                            overlap = iou(face.bbox, dilated)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_bbox = cand_bbox
                                best_offset = frame_offset
                        if best_bbox is not None:
                            match_meta[(i, j)] = {
                                "overlap": float(best_overlap),
                                "person_bbox": best_bbox,
                                "frame_offset": int(best_offset),
                            }
                        if best_overlap >= self.config.face_in_track_iou:
                            cost[i, j] = -best_overlap

                if np.any(cost < 1e6):
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for r, c in zip(row_ind, col_ind):
                        if cost[r, c] == 1e6:
                            continue
                        track_id = track_ids[c]
                        meta = match_meta.get((r, c))
                        if not meta:
                            continue
                        candidate = AssignedFace(
                            face_idx=r,
                            mode="iou",
                            overlap=float(meta["overlap"]),
                            score=float(face_dets[r].score),
                            person_bbox=tuple(meta["person_bbox"]),
                            frame_offset=meta["frame_offset"],
                        )
                        existing = assignments.get(track_id)
                        if not self.config.multi_face_per_track_guard or existing is None:
                            assignments[track_id] = candidate
                            unmatched_faces.discard(candidate.face_idx)
                        elif prefer_candidate(existing, candidate):
                            unmatched_faces.add(existing.face_idx)
                            assignments[track_id] = candidate
                            unmatched_faces.discard(candidate.face_idx)

            if face_dets and self.config.allow_face_center:
                for track_id, obs in track_lookup.items():
                    if track_id in assignments:
                        continue
                    candidate_faces = []
                    dilated_bbox = self._dilate_bbox(obs.bbox, self.config.dilate_track_px, width, height)
                    for face_idx in list(unmatched_faces):
                        if self._face_center_in_bbox(face_dets[face_idx].bbox, dilated_bbox):
                            candidate_faces.append(face_idx)
                    if len(candidate_faces) == 1:
                        face_idx = candidate_faces[0]
                        candidate = AssignedFace(
                            face_idx=face_idx,
                            mode="center",
                            overlap=0.0,
                            score=float(face_dets[face_idx].score),
                            person_bbox=obs.bbox,
                            frame_offset=0,
                        )
                        assignments[track_id] = candidate
                        unmatched_faces.discard(face_idx)

            for face_idx in list(unmatched_faces):
                face = face_dets[face_idx]
                best_track, best_iou, best_bbox = self._best_iou_track(
                    face.bbox,
                    track_lookup,
                    accumulator,
                    frame_idx,
                    width,
                    height,
                )
                record_reject(
                    best_track,
                    "rejected_low_iou",
                    frame_idx,
                    {
                        "best_iou": float(best_iou),
                        "face_bbox": list(map(float, face.bbox)),
                        "person_bbox": list(map(float, best_bbox)) if best_bbox else None,
                        "score": face.score,
                    },
                )

            for track_id, assignment in assignments.items():
                face = face_dets[assignment.face_idx]
                if not self._passes_area_threshold(face.bbox, frame_area):
                    record_reject(
                        track_id,
                        "rejected_area",
                        frame_idx,
                        {
                            "bbox_area": bbox_area(face.bbox),
                            "frame_area": frame_area,
                            "face_bbox": list(map(float, face.bbox)),
                            "person_bbox": list(map(float, assignment.person_bbox)),
                        },
                    )
                    continue

                aligned = self.face_detector.align_to_112(frame, face.landmarks, face.bbox)
                sharpness = self._compute_sharpness(aligned)

                # Identity embedding and harvest id mapping
                embedding = None
                try:
                    if self.embedder is not None:
                        embedding = self.embedder.embed(aligned)
                except Exception:
                    embedding = None

                byte_id = track_id
                hid = byte_to_harvest.get(byte_id)
                if hid is None:
                    hid = _new_harvest_id()
                    byte_to_harvest[byte_id] = hid
                    harvest_source_byte[hid] = byte_id

                if self.config.identity_guard and embedding is not None:
                    if harvest_count.get(hid, 0) >= self.config.identity_min_picks:
                        sim = _cosine(embedding, harvest_centroid[hid])
                        if sim < self.config.identity_sim_threshold:
                            if self.config.identity_split:
                                prev_hid = hid
                                hid = _new_harvest_id()
                                byte_to_harvest[byte_id] = hid
                                harvest_source_byte[hid] = byte_id
                                frame_events[frame_idx].append(
                                    {
                                        "frame": frame_idx,
                                        "event": "identity_split",
                                        "track_id": hid,
                                        "byte_track_id": byte_id,
                                        "prev_harvest_id": prev_hid,
                                        "cosine_sim": float(sim),
                                    }
                                )
                            else:
                                record_reject(
                                    byte_id,
                                    "rejected_identity_purity",
                                    frame_idx,
                                    {"harvest_id": hid, "cosine_sim": float(sim)},
                                )
                                continue

                harvest_dirname = (
                    f"track_{hid:04d}" if self.config.reindex_harvest_tracks else f"track_{byte_id:04d}"
                )
                track_dir = ensure_dir(crops_dir / harvest_dirname)
                sample_path = track_dir / f"{video_stem}_f{frame_idx:06d}.jpg"
                cv2.imwrite(str(sample_path), aligned)

                area = bbox_area(face.bbox)
                area_frac = area / frame_area if frame_area else 0.0
                orientation, frontalness = self._orientation_metrics(face.landmarks)
                quality = self._score_quality(sharpness, frontalness, area_frac)

                face_sample = FaceSample(
                    track_id=hid,
                    byte_track_id=byte_id,
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    path=sample_path,
                    score=face.score,
                    bbox=face.bbox,
                    quality=quality,
                    sharpness=sharpness,
                    orientation=orientation,
                    frontalness=frontalness,
                    area_frac=area_frac,
                    person_bbox=assignment.person_bbox,
                    association_iou=float(assignment.overlap),
                    match_mode=assignment.mode,
                    match_frame_offset=assignment.frame_offset,
                    embedding=embedding if embedding is not None else None,
                )

                if not self._passes_sharpness_threshold(sharpness):
                    record_reject(
                        track_id,
                        "rejected_sharpness",
                        frame_idx,
                        {
                            "sharpness": float(sharpness),
                            "face_bbox": list(map(float, face.bbox)),
                            "person_bbox": list(map(float, assignment.person_bbox)),
                        },
                    )
                    manage_sample_file(face_sample, "rejected_sharpness")
                    continue

                record_candidate_event(
                    hid,
                    frame_idx,
                    {
                        "reason": "candidate",
                        "byte_track_id": byte_id,
                        "face_bbox": list(map(float, face.bbox)),
                        "person_bbox": list(map(float, assignment.person_bbox)),
                        "association_iou": float(assignment.overlap),
                        "sharpness": float(sharpness),
                        "frontalness": float(frontalness),
                        "quality": float(quality),
                        "area_frac": float(area_frac),
                        "match_mode": assignment.mode,
                        "frame_offset": int(assignment.frame_offset),
                        "path": str(sample_path),
                    },
                )
                
                sampling_state = sampling_states.setdefault(hid, TrackSamplingState())
                removed = sampling_state.add_candidate(
                    face_sample,
                    quality,
                    sharpness,
                    frontalness,
                    area_frac,
                    orientation,
                    self.config,
                )
                if removed is not None:
                    removed_event = candidate_event_lookup.get((removed.track_id, removed.frame_idx))
                    if removed_event:
                        removed_event["reason"] = "rejected_quality"
                        removed_event["picked"] = False
                    record_reject(
                        removed.track_id,
                        "rejected_quality",
                        removed.frame_idx,
                        {
                            "quality": float(removed.quality),
                            "orientation": removed.orientation,
                            "face_bbox": list(map(float, removed.bbox)),
                            "person_bbox": list(map(float, removed.person_bbox)),
                        },
                    )
                    manage_sample_file(removed, "rejected_quality", removed_event)

                # Provisional centroid update to enable on-the-fly identity guard decisions
                if face_sample.embedding is not None:
                    if hid not in harvest_centroid:
                        harvest_centroid[hid] = face_sample.embedding.copy()
                        harvest_count[hid] = 1
                    else:
                        n = harvest_count.get(hid, 0)
                        harvest_centroid[hid] = (harvest_centroid[hid] * n + face_sample.embedding) / (n + 1)
                        harvest_count[hid] = n + 1

            log_progress(frames_seen)
            last_frames_seen = frames_seen

        if last_frames_seen:
            if last_frames_seen < next_progress_log:
                next_progress_log = last_frames_seen
            log_progress(last_frames_seen)

        cap.release()
        finished_tracks = accumulator.flush()

        selection_map: Dict[int, List[FaceSample]] = {}
        candidate_rows: List[Dict[str, Any]] = []
        for hid, state in sampling_states.items():
            selected = state.export_samples(self.config)
            selection_map[hid] = selected
            for candidate in state.candidates:
                event = candidate_event_lookup.get((hid, candidate.sample.frame_idx))
                if event:
                    event["reason"] = candidate.reason or ("picked" if candidate.picked else "not_selected")
                    event["picked"] = candidate.picked
                    event["quality"] = candidate.quality
                    event["sharpness"] = candidate.sharpness
                    event["frontalness"] = candidate.frontalness
                    event["area_frac"] = candidate.area_frac
                    event.setdefault("association_iou", candidate.sample.association_iou)
                if candidate.reason and candidate.reason.startswith("rejected_"):
                    reject_counter[candidate.reason] += 1
                if not candidate.picked:
                    manage_sample_file(candidate.sample, candidate.reason or "not_selected", event)
                else:
                    if event is not None:
                        event["path"] = str(candidate.sample.path)
        # Ensure keys present
        for hid in list(sampling_states.keys()):
            selection_map.setdefault(hid, [])

        for hid, state in sampling_states.items():
            candidate_rows.extend(state.iter_debug_rows(hid))

        # Update identity centroids using picked samples only
        for hid, selected in selection_map.items():
            for s in selected:
                if s.embedding is None:
                    continue
                if hid not in harvest_centroid:
                    harvest_centroid[hid] = s.embedding.copy()
                    harvest_count[hid] = 1
                else:
                    n = harvest_count[hid]
                    harvest_centroid[hid] = (harvest_centroid[hid] * n + s.embedding) / (n + 1)
                    harvest_count[hid] = n + 1

        # Build manifest per harvest id, carrying source ByteTrack id
        track_by_id: Dict[int, TrackState] = {t.track_id: t for t in finished_tracks}
        manifest_entries: List[ManifestEntry] = []
        for hid, samples in selection_map.items():
            byte_id = harvest_source_byte.get(hid)
            track_state = track_by_id.get(byte_id) if byte_id is not None else None
            entry = ManifestEntry(
                track_id=hid,
                byte_track_id=byte_id,
                total_frames=track_state.duration_frames if track_state else 0,
                avg_conf=track_state.avg_score if track_state else 0.0,
                avg_area=float(np.mean([bbox_area(b) for b in track_state.bboxes])) if track_state and track_state.bboxes else 0.0,
                first_ts_ms=track_state.timestamps_ms[0] if track_state and track_state.timestamps_ms else 0.0,
                last_ts_ms=track_state.timestamps_ms[-1] if track_state and track_state.timestamps_ms else 0.0,
                samples=samples,
            )
            manifest_entries.append(entry)

        manifest_path = harvest_dir / "manifest.json"
        dump_json(manifest_path, [entry.to_dict() for entry in manifest_entries])

        if candidate_rows:
            selected_csv = harvest_dir / "selected_samples.csv"
            fieldnames = [
                "track_id",
                "byte_track_id",
                "frame",
                "quality",
                "sharpness",
                "frontalness",
                "area_frac",
                "picked",
                "reason",
                "path",
                "association_iou",
                "match_mode",
                "frame_offset",
            ]
            with selected_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(candidate_rows)
            LOGGER.info("Harvest selection summary written to %s", selected_csv)

        if self.config.debug_rejections:
            debug_path = harvest_dir / "harvest_debug.json"
            payload: Dict[str, Any] = {
                "reject_counts": dict(reject_counter),
                "frame_events": {str(frame): events for frame, events in frame_events.items()},
            }
            if debug_records:
                payload["reject_details"] = {k: v for k, v in debug_records.items()}
            dump_json(debug_path, payload)
            LOGGER.info("Harvest debug log written to %s", debug_path)

        LOGGER.info(
            "Harvest complete tracks=%d samples=%d manifest=%s",
            len(manifest_entries),
            sum(len(entry.samples) for entry in manifest_entries),
            manifest_path,
        )
        return manifest_path

    def _passes_area_threshold(self, bbox, frame_area: float) -> bool:
        area = bbox_area(bbox)
        if self.config.min_area_px and area < self.config.min_area_px:
            return False
        if area / frame_area < self.config.min_area_frac:
            return False
        return True

    def _passes_sharpness_threshold(self, sharpness: float) -> bool:
        if self.config.min_sharpness_laplacian is None:
            return True
        return sharpness >= self.config.min_sharpness_laplacian

    def _build_manifest(
        self,
        tracks: List[TrackState],
        selection_map: Dict[int, List[FaceSample]],
    ) -> List[ManifestEntry]:
        manifest: List[ManifestEntry] = []
        for track in tracks:
            samples = selection_map.get(track.track_id, [])
            entry = ManifestEntry(
                track_id=track.track_id,
                total_frames=track.duration_frames,
                avg_conf=track.avg_score,
                avg_area=float(np.mean([bbox_area(b) for b in track.bboxes])) if track.bboxes else 0.0,
                first_ts_ms=track.timestamps_ms[0] if track.timestamps_ms else 0.0,
                last_ts_ms=track.timestamps_ms[-1] if track.timestamps_ms else 0.0,
                samples=samples,
            )
            manifest.append(entry)

        return manifest

    def _best_iou_track(
        self,
        face_bbox: Tuple[float, float, float, float],
        track_lookup: Dict[int, TrackObservation],
        accumulator: TrackAccumulator,
        frame_idx: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[Optional[int], float, Optional[Tuple[float, float, float, float]]]:
        best_track: Optional[int] = None
        best_iou = 0.0
        best_bbox: Optional[Tuple[float, float, float, float]] = None
        for track_id, obs in track_lookup.items():
            candidate_boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
            state = accumulator.active.get(track_id)
            tolerance = max(0, self.config.temporal_iou_tolerance)
            if state:
                for past_frame, bbox in zip(reversed(state.frames), reversed(state.bboxes)):
                    delta = frame_idx - past_frame
                    if delta < 0:
                        continue
                    if delta > tolerance:
                        break
                    candidate_boxes.append((delta, bbox))
            else:
                candidate_boxes.append((0, obs.bbox))
            for _, bbox in candidate_boxes:
                dilated = self._dilate_bbox(bbox, self.config.dilate_track_px, frame_width, frame_height)
                overlap = iou(face_bbox, dilated)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track = track_id
                    best_bbox = bbox
        return best_track, best_iou, best_bbox

    def _face_center_in_bbox(
        self,
        face_bbox: Tuple[float, float, float, float],
        person_bbox: Tuple[float, float, float, float],
    ) -> bool:
        x_center = (face_bbox[0] + face_bbox[2]) / 2.0
        y_center = (face_bbox[1] + face_bbox[3]) / 2.0
        return self._point_in_bbox((x_center, y_center), person_bbox)

    @staticmethod
    def _dilate_bbox(
        bbox: Tuple[float, float, float, float],
        dilation: float,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float, float]:
        if dilation <= 0:
            return bbox
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        dx = width * dilation
        dy = height * dilation
        nx1 = max(0.0, x1 - dx)
        ny1 = max(0.0, y1 - dy)
        nx2 = min(float(frame_width), x2 + dx)
        ny2 = min(float(frame_height), y2 + dy)
        return (nx1, ny1, nx2, ny2)

    def _compute_sharpness(self, aligned: np.ndarray) -> float:
        gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return variance

    def _score_quality(
        self,
        sharpness: float,
        frontalness: float,
        area_frac: float,
    ) -> float:
        w_sharp, w_front, w_area = self.config.quality_weights
        base_floor = self.config.min_sharpness_laplacian or 0.0
        sharp_scale = max(base_floor * 3.0, 1.0)
        sharp_norm = float(np.clip((sharpness - base_floor) / sharp_scale, 0.0, 1.0))
        frontal_score = float(np.clip(frontalness, 0.0, 1.0))
        area_score = self._area_closeness(area_frac)
        quality = (w_sharp * sharp_norm) + (w_front * frontal_score) + (w_area * area_score)
        return float(np.clip(quality, 0.0, 1.0))

    def _area_closeness(self, area_frac: float) -> float:
        target = max(self.config.target_area_frac, 1e-6)
        delta = abs(area_frac - target)
        closeness = 1.0 - (delta / target)
        return float(np.clip(closeness, 0.0, 1.0))

    def _orientation_metrics(self, landmarks: Optional[np.ndarray]) -> Tuple[str, float]:
        if landmarks is None:
            return "unknown", 0.4
        points = np.asarray(landmarks)
        if points.ndim != 2 or points.shape[0] < 2:
            return "unknown", 0.4
        left_eye = points[0]
        right_eye = points[1]
        nose = points[2] if points.shape[0] > 2 else (left_eye + right_eye) / 2.0
        dist_left = float(abs(nose[0] - left_eye[0]))
        dist_right = float(abs(right_eye[0] - nose[0]))
        denom = max(dist_left + dist_right, 1e-6)
        asymmetry = abs(dist_left - dist_right) / denom
        thresh = max(self.config.profile_asymmetry_thresh, 1e-6)
        label = "frontal" if asymmetry < thresh else "profile"
        ratio = asymmetry / (thresh * 1.5)
        frontalness = float(np.clip(1.0 - ratio, 0.0, 1.0))
        if label == "profile":
            frontalness *= 0.5
        return label, frontalness

    @staticmethod
    def _point_in_bbox(point: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> bool:
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
