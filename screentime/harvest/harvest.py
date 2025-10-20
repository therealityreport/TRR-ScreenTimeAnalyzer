"""Harvest pipeline orchestration."""

from __future__ import annotations

import csv
import logging
import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.io_utils import dump_json, ensure_dir, infer_video_stem
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper, TrackAccumulator, TrackObservation
from screentime.types import Detection, FaceSample, ManifestEntry, TrackState, bbox_area, iou
from screentime.recognition.embed_arcface import ArcFaceEmbedder

LOGGER = logging.getLogger("screentime.harvest")


def _flush_logger(logger: logging.Logger) -> None:
    """Best-effort flush of attached handlers to surface logs promptly."""
    for handler in logger.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # pragma: no cover - flushing is best effort
                continue


@dataclass
class HarvestConfig:
    stride: int = 1
    samples_min: int = 6
    samples_max: int = 12
    samples_per_track: int = 8
    min_gap_frames: int = 8
    min_area_frac: float = 0.005
    min_area_px: Optional[float] = None
    min_face_px: int = 0
    min_sharpness_laplacian: Optional[float] = None
    min_sharpness_pct: Optional[float] = None
    sharpness_pctile: Optional[float] = 40.0
    min_frontalness: float = 0.35
    frontal_pctile: Optional[float] = 50.0
    min_frontal_picks: int = 2
    face_in_track_iou: float = 0.25
    allow_face_center: bool = False
    dilate_track_px: float = 0.07
    temporal_iou_tolerance: int = 1
    profile_asymmetry_thresh: float = 0.35
    quality_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    target_area_frac: float = 0.02
    debug_rejections: bool = False
    multi_face_per_track_guard: bool = True
    multi_face_tiebreak: str = "quality"
    fallback_head_pct: float = 0.4
    # Identity purity guard across frames
    identity_guard: bool = True
    identity_split: bool = True
    identity_sim_threshold: float = 0.55
    identity_min_picks: int = 3
    identity_guard_stride: int = 6
    identity_guard_consecutive: int = 3
    identity_guard_cosine_reject: float = 0.35
    # Reindex harvest folders independent of ByteTrack ids
    reindex_harvest_tracks: bool = True
    fast_mode: bool = False
    min_track_frames: int = 12
    write_candidates: bool = False
    recall_pass: bool = True
    recall_det_thresh: float = 0.20
    recall_face_iou: float = 0.15
    recall_track_iou: float = 0.3
    recall_max_gap: int = 4
    progress_percent_interval: float = 5.0
    progress_fallback_frames: int = 500
    heartbeat_seconds: float = 2.0
    defer_embeddings: bool = False
    threads: int = 1
    new_track_min_frames: int = 3
    max_new_tracks_per_sec: float = 2.0
    stitch_identities: bool = True
    stitch_sim: float = 0.45
    stitch_gap_ms: float = 8000.0
    stitch_min_iou: float = 0.1


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
    cosine_similarity: Optional[float] = None


@dataclass
class DetectionContext:
    detection: Detection
    aligned: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    provider: Optional[str] = None
    identity_ready: bool = False


@dataclass
class PendingGuardState:
    samples: List[FaceSample] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    assignments: List[AssignedFace] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class TrackInfo:
    harvest_id: int
    byte_id: Optional[int]
    samples: List[FaceSample]
    track_state: Optional[TrackState]
    medoid: Optional[np.ndarray]
    start_ts_ms: float
    end_ts_ms: float
    start_bbox: Optional[Tuple[float, float, float, float]]
    end_bbox: Optional[Tuple[float, float, float, float]]
    total_frames: int
    avg_conf: float
    avg_area: float


@dataclass
class StitchedTrack:
    super_id: int
    source_ids: List[int]
    byte_ids: List[Optional[int]]
    samples: List[FaceSample]
    total_frames: int
    avg_conf: float
    avg_area: float
    start_ts_ms: float
    end_ts_ms: float


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
        pct_setting = config.sharpness_pctile
        if pct_setting is None:
            pct_setting = config.min_sharpness_pct
        if pct_setting is not None and sorted_candidates:
            sharp_thresholds.append(float(np.percentile([cand.sharpness for cand in sorted_candidates], pct_setting)))
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
            if sorted_candidates:
                fallback = sorted_candidates[0]
                fallback.picked = True
                fallback.reason = fallback.reason or "picked_low_quality"
                return [fallback.sample]
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

        frontal_threshold: Optional[float] = None
        if config.frontal_pctile is not None and eligible_candidates:
            frontal_threshold = float(
                np.percentile([cand.frontalness for cand in eligible_candidates], config.frontal_pctile)
            )

        if frontal_threshold is not None and config.min_frontal_picks > 0:
            frontal_pool = [cand for cand in eligible_candidates if cand.frontalness >= frontal_threshold]
            frontal_pool.sort(key=lambda c: c.quality, reverse=True)
            selected.sort(key=lambda c: c.quality, reverse=True)
            for cand in frontal_pool:
                if sum(1 for s in selected if s.frontalness >= frontal_threshold) >= config.min_frontal_picks:
                    break
                if cand in selected:
                    continue
                if len(selected) >= config.samples_per_track:
                    # replace lowest quality candidate that does not meet threshold
                    replace_target = None
                    for existing in reversed(sorted(selected, key=lambda c: c.quality)):
                        if existing.frontalness < frontal_threshold:
                            replace_target = existing
                            break
                    if replace_target is None:
                        continue
                    replace_target.picked = False
                    replace_target.reason = replace_target.reason or "not_selected"
                    selected.remove(replace_target)
                cand.picked = True
                cand.reason = "picked"
                selected.append(cand)

        # ensure ordering back to original quality sort
        selected.sort(key=lambda c: c.sample.frame_idx)

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
                "path": str(candidate.sample.path) if candidate.picked else "",
                "association_iou": candidate.sample.association_iou,
                "match_mode": candidate.sample.match_mode,
                "frame_offset": candidate.sample.match_frame_offset,
                "identity_cosine": candidate.sample.identity_cosine,
                "similarity_to_centroid": candidate.sample.similarity_to_centroid,
                "provider": candidate.sample.provider,
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
        need_embedder = self.config.identity_guard or not self.config.defer_embeddings
        if need_embedder and self.embedder is None:
            try:
                self.embedder = ArcFaceEmbedder(threads=self.config.threads)
            except Exception as exc:  # pragma: no cover - runtime guard
                LOGGER.warning("ArcFace embedder unavailable (%s); disabling identity_guard for this run", exc)
                self.embedder = None
                if self.config.identity_guard:
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
        harvest_dir = ensure_dir(output_root)
        crops_dir = harvest_dir
        candidates_root = harvest_dir / "candidates"
        if self.config.write_candidates:
            candidates_root = ensure_dir(candidates_root)
        candidate_dirs: Dict[int, Path] = {}

        def _candidate_dir(track_id: int) -> Path:
            if not self.config.write_candidates:
                return candidates_root / f"track_{track_id:04d}"
            track_dir = candidate_dirs.get(track_id)
            if track_dir is None:
                track_dir = ensure_dir(candidates_root / f"track_{track_id:04d}")
                candidate_dirs[track_id] = track_dir
            return track_dir

        def save_candidate_image(image: Optional[np.ndarray], track_id: int, frame_idx: int) -> None:
            if not self.config.write_candidates or image is None:
                return
            track_dir = _candidate_dir(track_id)
            candidate_path = track_dir / f"F{frame_idx:06d}.jpg"
            cv2.imwrite(str(candidate_path), image)

        accumulator = TrackAccumulator()
        sampling_states: Dict[int, TrackSamplingState] = {}
        selection_map: Dict[int, List[FaceSample]] = {}
        debug_records: Dict[str, List[Dict]] = defaultdict(list)
        reject_counter: Counter[str] = Counter()
        frame_events: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        candidate_event_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
        debug_write_queue: List[Tuple[np.ndarray, Path]] = []
        candidate_rows: List[Dict[str, Any]] = []

        # Identity guard state
        guard_centroid: Dict[int, np.ndarray] = {}
        guard_count: Dict[int, int] = {}
        guard_last_frame: Dict[int, int] = {}
        guard_pending: Dict[int, PendingGuardState] = {}
        guard_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        split_events: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        # Track lifecycle gating
        track_confirmed: Dict[int, bool] = {}
        track_pending_frames: Dict[int, int] = {}
        track_first_ts: Dict[int, float] = {}
        track_last_ts: Dict[int, float] = {}
        track_spawn_history: Deque[float] = deque()

        # Harvest track namespace mapping
        next_harvest_id = 1
        byte_to_harvest: Dict[int, int] = {}
        harvest_source_byte: Dict[int, Optional[int]] = {}
        recall_tracks: Dict[int, Dict[str, Any]] = {}

        def _new_harvest_id() -> int:
            nonlocal next_harvest_id
            hid = next_harvest_id
            next_harvest_id += 1
            return hid

        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            num = float((a * b).sum())
            den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            return num / den

        def _normalize(vec: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(vec)) + 1e-9
            return vec / norm

        def _can_spawn(timestamp_ms: float) -> bool:
            limit = float(self.config.max_new_tracks_per_sec)
            if limit <= 0:
                return True
            window_ms = 1000.0
            while track_spawn_history and (timestamp_ms - track_spawn_history[0]) > window_ms:
                track_spawn_history.popleft()
            return len(track_spawn_history) < limit

        def _register_spawn(timestamp_ms: float) -> None:
            track_spawn_history.append(timestamp_ms)

        def _record_guard_embedding(hid: int, embedding: np.ndarray, frame_idx: int, force: bool = False) -> None:
            guard_embeddings[hid].append(embedding)
            stride = max(1, int(self.config.identity_guard_stride))
            last_frame = guard_last_frame.get(hid, -10**9)
            should_update = force or (frame_idx - last_frame) >= stride or guard_count.get(hid, 0) == 0
            if not should_update:
                return
            if hid not in guard_centroid:
                guard_centroid[hid] = embedding.copy()
                guard_count[hid] = 1
            else:
                centroid = guard_centroid[hid] + embedding
                guard_centroid[hid] = _normalize(centroid)
                guard_count[hid] = guard_count.get(hid, 0) + 1
            guard_last_frame[hid] = frame_idx

        def _commit_sample(
            target_id: int,
            sample_obj: FaceSample,
            guard_embed: Optional[np.ndarray],
            assign_obj: AssignedFace,
            reason_label: str,
        ) -> None:
            selection_map.setdefault(target_id, [])
            sampling_state = sampling_states.setdefault(target_id, TrackSamplingState())
            record_candidate_event(
                target_id,
                sample_obj.frame_idx,
                {
                    "reason": reason_label,
                    "byte_track_id": sample_obj.byte_track_id,
                    "face_bbox": list(map(float, sample_obj.bbox)),
                    "person_bbox": list(map(float, assign_obj.person_bbox)),
                    "association_iou": float(assign_obj.overlap),
                    "sharpness": float(sample_obj.sharpness),
                    "frontalness": float(sample_obj.frontalness),
                    "quality": float(sample_obj.quality),
                    "area_frac": float(sample_obj.area_frac),
                    "match_mode": assign_obj.mode,
                    "frame_offset": int(assign_obj.frame_offset),
                    "path": str(sample_obj.path),
                    "identity_cosine": sample_obj.identity_cosine,
                    "provider": sample_obj.provider,
                    "cosine_similarity": assign_obj.cosine_similarity,
                },
            )
            save_candidate_image(sample_obj.image, target_id, sample_obj.frame_idx)
            removed = sampling_state.add_candidate(
                sample_obj,
                sample_obj.quality,
                sample_obj.sharpness,
                sample_obj.frontalness,
                sample_obj.area_frac,
                sample_obj.orientation,
                self.config,
            )
            if removed is not None:
                removed_event = candidate_event_lookup.get((removed.track_id, removed.frame_idx))
                if removed_event:
                    removed_event["reason"] = "rejected_quality"
                    removed_event["picked"] = False
            if self.config.identity_guard and guard_embed is not None:
                force_update = guard_count.get(target_id, 0) == 0
                _record_guard_embedding(target_id, guard_embed, sample_obj.frame_idx, force=force_update)

        def _compute_medoid_embedding(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
            if not embeddings:
                return None
            if len(embeddings) == 1:
                return embeddings[0].copy()
            arr = np.stack(embeddings, axis=0)
            sims = np.clip(arr @ arr.T, -1.0, 1.0)
            distances = 1.0 - sims
            mean_dist = distances.mean(axis=1)
            idx = int(np.argmin(mean_dist))
            return arr[idx].copy()

        def _build_track_infos() -> Dict[int, TrackInfo]:
            infos: Dict[int, TrackInfo] = {}
            for hid, samples in selection_map.items():
                if not samples:
                    continue
                byte_id = harvest_source_byte.get(hid)
                track_state = track_by_id.get(byte_id) if byte_id is not None else None
                medoid = _compute_medoid_embedding(guard_embeddings.get(hid, []))
                if track_state and track_state.timestamps_ms:
                    start_ts = float(track_state.timestamps_ms[0])
                    end_ts = float(track_state.timestamps_ms[-1])
                else:
                    timestamps = [float(sample.timestamp_ms) for sample in samples if sample.timestamp_ms is not None]
                    if timestamps:
                        start_ts = min(timestamps)
                        end_ts = max(timestamps)
                    else:
                        start_ts = 0.0
                        end_ts = 0.0
                if track_state and track_state.bboxes:
                    start_bbox = track_state.bboxes[0]
                    end_bbox = track_state.bboxes[-1]
                else:
                    start_bbox = samples[0].person_bbox if samples else None
                    end_bbox = samples[-1].person_bbox if samples else None
                if track_state and track_state.scores:
                    avg_conf = float(np.mean(track_state.scores))
                else:
                    avg_conf = float(np.mean([sample.score for sample in samples])) if samples else 0.0
                if track_state and track_state.bboxes:
                    avg_area = float(np.mean([bbox_area(bbox) for bbox in track_state.bboxes]))
                else:
                    avg_area = float(np.mean([bbox_area(sample.person_bbox) for sample in samples])) if samples else 0.0
                total_frames = int(track_state.duration_frames) if track_state else len(samples)
                infos[hid] = TrackInfo(
                    harvest_id=hid,
                    byte_id=byte_id,
                    samples=samples,
                    track_state=track_state,
                    medoid=medoid,
                    start_ts_ms=start_ts,
                    end_ts_ms=end_ts,
                    start_bbox=start_bbox,
                    end_bbox=end_bbox,
                    total_frames=total_frames,
                    avg_conf=avg_conf,
                    avg_area=avg_area,
                )
            return infos

        def _stitch_track_infos(track_infos: Dict[int, TrackInfo]) -> List[StitchedTrack]:
            if not track_infos:
                return []

            sorted_ids = sorted(track_infos.keys(), key=lambda tid: track_infos[tid].start_ts_ms)

            if not self.config.stitch_identities:
                stitched: List[StitchedTrack] = []
                for super_idx, hid in enumerate(sorted_ids, start=1):
                    info = track_infos[hid]
                    byte_ids = [info.byte_id] if info.byte_id is not None else []
                    stitched.append(
                        StitchedTrack(
                            super_id=super_idx,
                            source_ids=[hid],
                            byte_ids=byte_ids,
                            samples=info.samples,
                            total_frames=info.total_frames,
                            avg_conf=info.avg_conf,
                            avg_area=info.avg_area,
                            start_ts_ms=info.start_ts_ms,
                            end_ts_ms=info.end_ts_ms,
                        )
                    )
                return stitched

            medoid_ids = [hid for hid, info in track_infos.items() if info.medoid is not None]
            parent: Dict[int, int] = {hid: hid for hid in medoid_ids}

            def find_root(x: int) -> int:
                root = x
                while parent[root] != root:
                    root = parent[root]
                # Path compression
                while parent[x] != x:
                    nxt = parent[x]
                    parent[x] = root
                    x = nxt
                return root

            def union(a: int, b: int) -> None:
                ra = find_root(a)
                rb = find_root(b)
                if ra != rb:
                    parent[rb] = ra

            for idx_a, hid_a in enumerate(medoid_ids):
                info_a = track_infos[hid_a]
                for hid_b in medoid_ids[idx_a + 1 :]:
                    info_b = track_infos[hid_b]
                    if info_a.medoid is None or info_b.medoid is None:
                        continue
                    sim_val = float(np.clip(np.dot(info_a.medoid, info_b.medoid), -1.0, 1.0))
                    distance = 1.0 - sim_val
                    if distance <= float(self.config.stitch_sim):
                        union(hid_a, hid_b)

            cluster_map: Dict[int, List[int]] = {}
            for hid in medoid_ids:
                root = find_root(hid)
                cluster_map.setdefault(root, []).append(hid)

            for hid in track_infos:
                if hid not in cluster_map and track_infos[hid].medoid is None:
                    cluster_map[hid] = [hid]

            clusters = []
            seen: set[int] = set()
            for members in cluster_map.values():
                ordered = sorted(members, key=lambda tid: track_infos[tid].start_ts_ms)
                clusters.append(ordered)
                seen.update(ordered)
            for hid in track_infos:
                if hid not in seen:
                    clusters.append([hid])

            clusters.sort(key=lambda ids: min(track_infos[tid].start_ts_ms for tid in ids))

            stitched: List[StitchedTrack] = []
            next_super_id = 1
            for cluster_idx, members in enumerate(clusters, start=1):
                groups: List[List[int]] = []
                current_group: List[int] = [members[0]]
                for tid in members[1:]:
                    prev_info = track_infos[current_group[-1]]
                    curr_info = track_infos[tid]
                    gap_ms = max(0.0, curr_info.start_ts_ms - prev_info.end_ts_ms)
                    prev_bbox = prev_info.end_bbox or (prev_info.samples[-1].person_bbox if prev_info.samples else None)
                    curr_bbox = curr_info.start_bbox or (curr_info.samples[0].person_bbox if curr_info.samples else None)
                    overlap = iou(prev_bbox, curr_bbox) if prev_bbox and curr_bbox else 0.0
                    sim_val = float("nan")
                    if prev_info.medoid is not None and curr_info.medoid is not None:
                        sim_raw = float(np.clip(np.dot(prev_info.medoid, curr_info.medoid), -1.0, 1.0))
                        sim_val = sim_raw
                    if gap_ms <= float(self.config.stitch_gap_ms) and overlap >= float(self.config.stitch_min_iou):
                        current_group.append(tid)
                        LOGGER.info(
                            "STITCH cluster %02d merge %04d+%04d gap=%.1fs sim=%.2f",
                            cluster_idx,
                            current_group[-2],
                            tid,
                            gap_ms / 1000.0,
                            sim_val,
                        )
                    else:
                        groups.append(current_group)
                        current_group = [tid]
                groups.append(current_group)

                for group in groups:
                    group_infos = [track_infos[tid] for tid in group]
                    combined_samples: List[FaceSample] = []
                    byte_set: set[int] = set()
                    byte_ids: List[Optional[int]] = []
                    total_frames = 0
                    conf_weighted = 0.0
                    area_weighted = 0.0
                    weight_sum = 0
                    start_ts = min(info.start_ts_ms for info in group_infos)
                    end_ts = max(info.end_ts_ms for info in group_infos)
                    for info in group_infos:
                        combined_samples.extend(info.samples)
                        if info.byte_id is not None and info.byte_id not in byte_set:
                            byte_set.add(info.byte_id)
                            byte_ids.append(info.byte_id)
                        total_frames += info.total_frames
                        conf_weighted += info.avg_conf * info.total_frames
                        area_weighted += info.avg_area * info.total_frames
                        weight_sum += info.total_frames
                    avg_conf = conf_weighted / weight_sum if weight_sum else 0.0
                    avg_area = area_weighted / weight_sum if weight_sum else 0.0
                    combined_samples.sort(key=lambda sample: (sample.timestamp_ms, sample.frame_idx))
                    stitched.append(
                        StitchedTrack(
                            super_id=next_super_id,
                            source_ids=group,
                            byte_ids=byte_ids,
                            samples=combined_samples,
                            total_frames=total_frames,
                            avg_conf=avg_conf,
                            avg_area=avg_area,
                            start_ts_ms=start_ts,
                            end_ts_ms=end_ts,
                        )
                    )
                    next_super_id += 1

            stitched.sort(key=lambda track: track.start_ts_ms)
            for new_idx, track in enumerate(stitched, start=1):
                track.super_id = new_idx
            return stitched

        def _build_detection_context(face_det: Detection) -> DetectionContext:
            aligned: Optional[np.ndarray] = None
            embedding: Optional[np.ndarray] = None
            provider: Optional[str] = None
            identity_ready = False
            try:
                aligned = face_ctx.aligned
                if aligned is None:
                    rebuilt = _build_detection_context(face_det)
                    if rebuilt.aligned is not None:
                        aligned = rebuilt.aligned
                        face_ctx.aligned = rebuilt.aligned
                    if face_ctx.embedding is None and rebuilt.embedding is not None:
                        face_ctx.embedding = rebuilt.embedding
                        face_ctx.provider = rebuilt.provider
                if aligned is None:
                    record_reject(
                        hid,
                        "rejected_alignment",
                        frame_idx,
                        {
                            "byte_track_id": byte_id,
                            "face_bbox": list(map(float, face_det.bbox)),
                        },
                    )
                    queue_debug_image(None, debug_path)
                    return False, hid
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.debug("Alignment failed for frame %s: %s", face_det.frame_idx, exc)
                aligned = None
            if (
                aligned is not None
                and self.config.identity_guard
                and self.embedder is not None
            ):
                try:
                    embedding = self.embedder.embed(aligned)
                    embedding = _normalize(embedding)
                    provider = getattr(self.embedder, "backend", None)
                    if not provider and hasattr(self.embedder, "providers"):
                        providers = list(getattr(self.embedder, "providers"))
                        provider = providers[0] if providers else None
                    identity_ready = True
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Identity guard embedding failed at frame %s: %s", face_det.frame_idx, exc)
                    embedding = None
            return DetectionContext(
                detection=face_det,
                aligned=aligned,
                embedding=embedding,
                provider=provider,
                identity_ready=identity_ready,
            )

        manifest_path = harvest_dir / "manifest.json"
        progress_path = harvest_dir / "progress.json"

        stub_manifest = {"version": 1, "tracks": [], "frames_total": int(frame_count or 0)}
        dump_json(manifest_path, stub_manifest)

        def current_track_count() -> int:
            return max(0, next_harvest_id - 1)

        def write_progress(frames_seen: int, track_count: int) -> None:
            percent_value = (frames_seen / frame_count * 100.0) if frame_count else None
            payload = {
                "frames_total": int(frame_count or 0),
                "frames_done": int(frames_seen),
                "percent": round(percent_value, 2) if percent_value is not None else None,
                "tracks": int(track_count),
                "last_update": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            dump_json(progress_path, payload)

        percent_interval = max(0.1, float(self.config.progress_percent_interval))
        fallback_frames = max(1, int(self.config.progress_fallback_frames))
        heartbeat_seconds = max(0.0, float(self.config.heartbeat_seconds))
        percent_frame_step: Optional[int] = None
        next_percent_mark: Optional[int] = None
        if frame_count:
            percent_frame_step = max(1, int(frame_count * (percent_interval / 100.0)))
            next_percent_mark = percent_frame_step
        last_progress_frames = 0
        last_heartbeat_ts = time.monotonic()

        def samples_total() -> int:
            return sum(len(state.candidates) for state in sampling_states.values())

        write_progress(0, current_track_count())

        frame_idx = -1
        processed_frames = 0

        LOGGER.info(
            "Starting harvest video=%s fps=%.2f frames=%s output=%s",
            video_path,
            fps,
            frame_count or "unknown",
            harvest_dir,
        )
        _flush_logger(LOGGER)

        def emit_status(frames_seen: int, processed: int, heartbeat: bool) -> None:
            track_count = current_track_count()
            percent_value = (frames_seen / frame_count * 100.0) if frame_count else None
            total_samples = samples_total()
            if heartbeat:
                if percent_value is not None:
                    LOGGER.info(
                        "Harvest heartbeat %.1f%% (%d/%d frames seen, processed=%d, active_tracks=%d, samples=%d)",
                        percent_value,
                        frames_seen,
                        frame_count,
                        processed,
                        len(accumulator.active),
                        total_samples,
                    )
                else:
                    LOGGER.info(
                        "Harvest heartbeat: %d frames seen (processed=%d, active_tracks=%d, samples=%d)",
                        frames_seen,
                        processed,
                        len(accumulator.active),
                        total_samples,
                    )
            else:
                if percent_value is not None:
                    LOGGER.info(
                        "Harvest progress %.1f%% (%d/%d frames seen, processed=%d, active_tracks=%d, samples=%d)",
                        percent_value,
                        frames_seen,
                        frame_count,
                        processed,
                        len(accumulator.active),
                        total_samples,
                    )
                else:
                    LOGGER.info(
                        "Harvest progress: %d frames seen (processed=%d, active_tracks=%d, samples=%d)",
                        frames_seen,
                        processed,
                        len(accumulator.active),
                        total_samples,
                    )
            write_progress(frames_seen, track_count)
            _flush_logger(LOGGER)

        def maybe_emit_status(frames_seen: int, processed: int) -> None:
            nonlocal next_percent_mark, last_progress_frames, last_heartbeat_ts
            now = time.monotonic()
            emit_progress = False
            if (
                next_percent_mark is not None
                and percent_frame_step is not None
                and frames_seen >= next_percent_mark
            ):
                emit_progress = True
                while (
                    next_percent_mark is not None
                    and percent_frame_step is not None
                    and frames_seen >= next_percent_mark
                ):
                    next_percent_mark += percent_frame_step
            if not emit_progress and frames_seen >= last_progress_frames + fallback_frames:
                emit_progress = True
            if emit_progress:
                last_progress_frames = frames_seen
                emit_status(frames_seen, processed, heartbeat=False)
                last_heartbeat_ts = now
                return
            if heartbeat_seconds > 0 and now - last_heartbeat_ts >= heartbeat_seconds:
                emit_status(frames_seen, processed, heartbeat=True)
                last_heartbeat_ts = now

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

        def queue_debug_image(image: Optional[np.ndarray], path: Path) -> None:
            if not self.config.debug_rejections or image is None:
                return
            debug_write_queue.append((image.copy(), path))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            frames_seen = frame_idx + 1
            maybe_emit_status(frames_seen, processed_frames)
            if frame_idx % self.config.stride != 0:
                continue

            processed_frames += 1
            timestamp_ms = (frame_idx / fps) * 1000.0

            person_dets = self.person_detector.detect(frame, frame_idx)
            observations = self.tracker.update(person_dets, frame.shape)
            accumulator.update(frame_idx, timestamp_ms, observations)
            active_ids = set(accumulator.active.keys())
            for tid in list(track_pending_frames.keys()):
                if tid not in active_ids:
                    track_pending_frames.pop(tid, None)
                    track_first_ts.pop(tid, None)
                    track_last_ts.pop(tid, None)
                    track_confirmed.pop(tid, None)
            track_lookup: Dict[int, TrackObservation] = {}
            for obs in observations:
                track_id = obs.track_id
                state = accumulator.active.get(track_id)
                if state is None:
                    continue
                if state.timestamps_ms:
                    last_ts = state.timestamps_ms[-1]
                    first_ts = state.timestamps_ms[0]
                else:
                    last_ts = timestamp_ms
                    first_ts = timestamp_ms
                track_first_ts.setdefault(track_id, first_ts)
                track_last_ts[track_id] = last_ts
                track_pending_frames[track_id] = state.duration_frames
                confirmed = track_confirmed.get(track_id, False)
                min_frames = max(1, int(self.config.new_track_min_frames))
                if not confirmed and state.duration_frames >= min_frames:
                    if _can_spawn(last_ts):
                        _register_spawn(last_ts)
                        track_confirmed[track_id] = True
                        confirmed = True
                    else:
                        track_confirmed.setdefault(track_id, False)
                if confirmed:
                    track_lookup[track_id] = obs

            face_dets = self.face_detector.detect(frame, frame_idx)
            face_contexts: Dict[int, DetectionContext] = {}
            if face_dets:
                for idx, face in enumerate(face_dets):
                    face_contexts[idx] = _build_detection_context(face)
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
                    ctx = face_contexts.get(i)
                    ctx_embedding = ctx.embedding if ctx else None
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
                        cosine: Optional[float] = None
                        if best_bbox is not None:
                            hid = byte_to_harvest.get(track_id)
                            centroid = guard_centroid.get(hid) if hid is not None else None
                            if centroid is not None and ctx_embedding is not None:
                                cosine = _cosine(ctx_embedding, centroid)
                            match_meta[(i, j)] = {
                                "overlap": float(best_overlap),
                                "person_bbox": best_bbox,
                                "frame_offset": int(best_offset),
                                "cosine": float(cosine) if cosine is not None else None,
                            }
                        if best_overlap >= self.config.face_in_track_iou and best_bbox is not None:
                            if cosine is not None and cosine < self.config.identity_guard_cosine_reject:
                                continue
                            base_cost = 1.0 - best_overlap
                            if cosine is not None:
                                identity_component = 1.0 - cosine
                                cost[i, j] = float(0.6 * base_cost + 0.4 * identity_component)
                            else:
                                cost[i, j] = float(base_cost)

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
                            cosine_similarity=float(meta["cosine"]) if meta.get("cosine") is not None else None,
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
                            cosine_similarity=None,
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

            def emit_sample(
                harvest_id: int,
                byte_id: Optional[int],
                assignment: AssignedFace,
                face_ctx: DetectionContext,
                reason: str,
            ) -> Tuple[bool, int]:
                nonlocal guard_centroid, guard_count
                hid = harvest_id
                face_det = face_ctx.detection

                sample_filename = f"{video_stem}_f{frame_idx:06d}.jpg"

                def resolve_paths(target_hid: int, source_byte: Optional[int]) -> Tuple[Path, Path]:
                    harvest_dirname_local = (
                        f"track_{target_hid:04d}"
                        if self.config.reindex_harvest_tracks or source_byte is None
                        else f"track_{source_byte:04d}"
                    )
                    sample_path_local = crops_dir / harvest_dirname_local / sample_filename
                    debug_path_local = sample_path_local.parent / "debug" / sample_filename
                    return sample_path_local, debug_path_local

                sample_path, debug_path = resolve_paths(hid, byte_id)

                if not self._passes_area_threshold(face_det.bbox, frame_area):
                    record_reject(
                        hid,
                        "rejected_area",
                        frame_idx,
                        {
                            "bbox_area": bbox_area(face_det.bbox),
                            "frame_area": frame_area,
                            "face_bbox": list(map(float, face_det.bbox)),
                            "person_bbox": list(map(float, assignment.person_bbox)),
                            "byte_track_id": byte_id,
                        },
                    )
                    return False, hid

                if not self._passes_face_size(face_det.bbox):
                    record_reject(
                        hid,
                        "rejected_face_size",
                        frame_idx,
                        {
                            "face_bbox": list(map(float, face_det.bbox)),
                            "min_face_px": self.config.min_face_px,
                            "byte_track_id": byte_id,
                        },
                    )
                    queue_debug_image(None, debug_path)
                    return False, hid

                aligned = face_ctx.aligned
                if aligned is None:
                    rebuilt = _build_detection_context(face_det)
                    if rebuilt.aligned is not None:
                        aligned = rebuilt.aligned
                        face_ctx.aligned = rebuilt.aligned
                    if face_ctx.embedding is None and rebuilt.embedding is not None:
                        face_ctx.embedding = rebuilt.embedding
                        face_ctx.provider = rebuilt.provider
                if aligned is None:
                    record_reject(
                        hid,
                        "rejected_alignment",
                        frame_idx,
                        {
                            "face_bbox": list(map(float, face_det.bbox)),
                            "byte_track_id": byte_id,
                        },
                    )
                    queue_debug_image(None, debug_path)
                    return False, hid

                sharpness = self._compute_sharpness(aligned)
                area = bbox_area(face_det.bbox)
                area_frac = area / frame_area if frame_area else 0.0
                orientation, frontalness = self._orientation_metrics(face_det.landmarks)
                quality = self._score_quality(sharpness, frontalness, area_frac)

                if frontalness < self.config.min_frontalness:
                    record_reject(
                        hid,
                        "rejected_frontalness",
                        frame_idx,
                        {
                            "frontalness": float(frontalness),
                            "min_frontalness": float(self.config.min_frontalness),
                            "byte_track_id": byte_id,
                        },
                    )
                    queue_debug_image(aligned, debug_path)
                    return False, hid

                if not self._passes_sharpness_threshold(sharpness):
                    record_reject(
                        hid,
                        "rejected_sharpness",
                        frame_idx,
                        {
                            "sharpness": float(sharpness),
                            "min_sharpness": self.config.min_sharpness_laplacian,
                            "byte_track_id": byte_id,
                        },
                    )
                    queue_debug_image(aligned, debug_path)
                    return False, hid

                timestamp_ms = frame_idx / fps * 1000.0 if fps else 0.0
                face_sample = FaceSample(
                    track_id=hid,
                    byte_track_id=byte_id,
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    path=sample_path,
                    score=face_det.score,
                    bbox=face_det.bbox,
                    quality=quality,
                    sharpness=sharpness,
                    orientation=orientation,
                    frontalness=frontalness,
                    area_frac=area_frac,
                    person_bbox=assignment.person_bbox,
                    association_iou=float(assignment.overlap),
                    match_mode=assignment.mode,
                    match_frame_offset=assignment.frame_offset,
                    image=aligned,
                )

                embed_provider = face_ctx.provider
                raw_embedding = face_ctx.embedding.copy() if face_ctx.embedding is not None else None
                guard_embedding = face_ctx.embedding
                if self.embedder is not None and guard_embedding is None:
                    try:
                        raw_embedding = self.embedder.embed(aligned)
                        guard_embedding = _normalize(raw_embedding)
                        face_ctx.embedding = guard_embedding
                        if embed_provider is None:
                            embed_provider = getattr(self.embedder, "backend", None)
                            if not embed_provider and hasattr(self.embedder, "providers"):
                                providers = list(getattr(self.embedder, "providers"))
                                embed_provider = providers[0] if providers else None
                        face_ctx.provider = embed_provider
                    except Exception as exc:  # pragma: no cover - defensive runtime guard
                        LOGGER.warning("Identity guard embedding failed for track %s frame %s: %s", hid, frame_idx, exc)
                        raw_embedding = None
                        guard_embedding = None
                if not self.config.defer_embeddings and raw_embedding is not None:
                    face_sample.embedding = raw_embedding.copy()
                else:
                    face_sample.embedding = None
                face_sample.provider = embed_provider

                committed: List[Dict[str, Any]] = []

                def queue_commit(
                    target_id: int,
                    sample_obj: FaceSample,
                    guard_embed: Optional[np.ndarray],
                    assign_obj: AssignedFace,
                    event_reason: str,
                ) -> None:
                    sample_path_final, _ = resolve_paths(target_id, byte_id)
                    ensure_dir(sample_path_final.parent)
                    sample_obj.track_id = target_id
                    sample_obj.path = sample_path_final
                    committed.append(
                        {
                            "target_id": target_id,
                            "sample": sample_obj,
                            "embedding": guard_embed,
                            "assignment": assign_obj,
                            "reason": event_reason,
                        }
                    )

                violation_threshold = float(self.config.identity_sim_threshold)
                sim: Optional[float] = None
                committed_current = False
                guard_ready = self.config.identity_guard and guard_embedding is not None
                if guard_ready:
                    centroid = guard_centroid.get(hid)
                    if centroid is not None and guard_count.get(hid, 0) >= self.config.identity_min_picks:
                        sim = _cosine(guard_embedding, centroid)
                        face_sample.identity_cosine = sim
                        if assignment.cosine_similarity is None:
                            assignment.cosine_similarity = sim
                        if sim < violation_threshold:
                            if not self.config.identity_split:
                                record_reject(
                                    hid,
                                    "rejected_identity_purity",
                                    frame_idx,
                                    {
                                        "byte_track_id": byte_id,
                                        "cosine_sim": float(sim),
                                        "threshold": violation_threshold,
                                    },
                                )
                                queue_debug_image(aligned, debug_path)
                                return False, hid
                            pending = guard_pending.get(hid)
                            if pending is None:
                                pending = PendingGuardState()
                                guard_pending[hid] = pending
                            pending.samples.append(face_sample)
                            pending.embeddings.append(guard_embedding)
                            pending.distances.append(1.0 - sim)
                            pending.assignments.append(
                                AssignedFace(
                                    face_idx=assignment.face_idx,
                                    mode=assignment.mode,
                                    overlap=assignment.overlap,
                                    score=assignment.score,
                                    person_bbox=assignment.person_bbox,
                                    frame_offset=assignment.frame_offset,
                                    cosine_similarity=assignment.cosine_similarity,
                                )
                            )
                            pending.reasons.append(reason)
                            consecutive = max(1, int(self.config.identity_guard_consecutive))
                            if len(pending.samples) >= consecutive:
                                new_hid = _new_harvest_id()
                                if byte_id is not None:
                                    byte_to_harvest[byte_id] = new_hid
                                harvest_source_byte[new_hid] = byte_id
                                payload = {
                                    "event": "identity_split",
                                    "track_id": new_hid,
                                    "byte_track_id": byte_id,
                                    "prev_harvest_id": harvest_id,
                                    "cosine_sim": float(sim),
                                }
                                log_frame_event(frame_idx, payload)
                                if byte_id is not None:
                                    split_events[byte_id].append({"frame": frame_idx, **payload})
                                LOGGER.info(
                                    "SPLIT track %04d at f=%d cos=%.2f",
                                    byte_id if byte_id is not None else harvest_id,
                                    frame_idx,
                                    float(sim),
                                )
                                for pending_sample, pending_embed, pending_assignment, pending_reason in zip(
                                    pending.samples,
                                    pending.embeddings,
                                    pending.assignments,
                                    pending.reasons,
                                ):
                                    queue_commit(new_hid, pending_sample, pending_embed, pending_assignment, pending_reason)
                                guard_pending.pop(harvest_id, None)
                                hid = new_hid
                                committed_current = True
                            else:
                                return False, hid
                    else:
                        face_sample.identity_cosine = sim
                        pending_existing = guard_pending.pop(hid, None)
                        if pending_existing:
                            for pending_sample, pending_embed, pending_assignment, pending_reason in zip(
                                pending_existing.samples,
                                pending_existing.embeddings,
                                pending_existing.assignments,
                                pending_existing.reasons,
                            ):
                                queue_commit(hid, pending_sample, pending_embed, pending_assignment, pending_reason)
                else:
                    face_sample.identity_cosine = sim
                    pending_existing = guard_pending.pop(hid, None)
                    if pending_existing:
                        for pending_sample, pending_embed, pending_assignment, pending_reason in zip(
                            pending_existing.samples,
                            pending_existing.embeddings,
                            pending_existing.assignments,
                            pending_existing.reasons,
                        ):
                            queue_commit(hid, pending_sample, pending_embed, pending_assignment, pending_reason)

                if not committed_current:
                    queue_commit(hid, face_sample, guard_embedding, assignment, reason)

                for payload in committed:
                    _commit_sample(
                        payload["target_id"],
                        payload["sample"],
                        payload["embedding"],
                        payload["assignment"],
                        payload["reason"],
                    )

                return True, hid

            for byte_id, assignment in assignments.items():
                hid = byte_to_harvest.get(byte_id)
                if hid is None:
                    hid = _new_harvest_id()
                    byte_to_harvest[byte_id] = hid
                    harvest_source_byte[hid] = byte_id
                else:
                    harvest_source_byte.setdefault(hid, byte_id)

                face_ctx = face_contexts.get(assignment.face_idx)
                if face_ctx is None:
                    face_ctx = _build_detection_context(face_dets[assignment.face_idx])
                    face_contexts[assignment.face_idx] = face_ctx
                success, new_hid = emit_sample(hid, byte_id, assignment, face_ctx, "candidate")
                if success:
                    harvest_source_byte[new_hid] = byte_id
                if new_hid != hid and byte_id is not None:
                    byte_to_harvest[byte_id] = new_hid
                if not success:
                    continue

            if self.config.recall_pass:
                missing_tracks = {tid for tid in track_lookup if tid not in assignments}
                recall_faces = []
                base_thresh = getattr(self.face_detector, "det_thresh", None)
                try:
                    if hasattr(self.face_detector, "det_thresh"):
                        self.face_detector.det_thresh = self.config.recall_det_thresh
                    recall_faces = self.face_detector.detect(frame, frame_idx)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Recall pass detection failed at frame %s: %s", frame_idx, exc)
                    recall_faces = []
                finally:
                    if base_thresh is not None and hasattr(self.face_detector, "det_thresh"):
                        self.face_detector.det_thresh = base_thresh

                if recall_faces:
                    assigned_boxes = [
                        face_dets[val.face_idx].bbox for val in assignments.values() if val.face_idx < len(face_dets)
                    ]
                    for face in recall_faces:
                        duplicate = False
                        for bbox in assigned_boxes:
                            if iou(face.bbox, bbox) >= 0.6:
                                duplicate = True
                                break
                        if duplicate:
                            continue

                        best_track, best_iou, best_bbox = self._best_iou_track(
                            face.bbox,
                            track_lookup,
                            accumulator,
                            frame_idx,
                            width,
                            height,
                        )

                        if best_track is not None and (
                            best_track in missing_tracks or best_iou >= self.config.recall_face_iou
                        ):
                            harvest_id = byte_to_harvest.get(best_track)
                            if harvest_id is None:
                                harvest_id = _new_harvest_id()
                                byte_to_harvest[best_track] = harvest_id
                                harvest_source_byte[harvest_id] = best_track
                            person_bbox = best_bbox if best_bbox is not None else track_lookup[best_track].bbox
                            assignment = AssignedFace(
                                face_idx=-1,
                                mode="recall_track",
                                overlap=float(best_iou),
                                score=float(face.score),
                                person_bbox=person_bbox,
                                frame_offset=0,
                            )
                            recall_ctx = _build_detection_context(face)
                            success, new_hid = emit_sample(harvest_id, best_track, assignment, recall_ctx, "recall")
                            if success:
                                missing_tracks.discard(best_track)
                                harvest_source_byte[new_hid] = best_track
                                if new_hid != harvest_id:
                                    byte_to_harvest[best_track] = new_hid
                            continue

                        # fall back to creating/merging recall-only track
                        best_recall_id = None
                        best_recall_iou = 0.0
                        for hid, state in recall_tracks.items():
                            gap = frame_idx - state["last_frame"]
                            if gap < 0 or gap > self.config.recall_max_gap:
                                continue
                            overlap = iou(face.bbox, state["bbox"])
                            if overlap > best_recall_iou:
                                best_recall_iou = overlap
                                best_recall_id = hid

                        if best_recall_id is not None and best_recall_iou >= self.config.recall_track_iou:
                            hid = best_recall_id
                        else:
                            hid = _new_harvest_id()
                            recall_tracks[hid] = {"last_frame": frame_idx, "bbox": face.bbox}
                            harvest_source_byte[hid] = None

                        recall_tracks[hid]["last_frame"] = frame_idx
                        recall_tracks[hid]["bbox"] = face.bbox

                        assignment = AssignedFace(
                            face_idx=-1,
                            mode="recall",
                            overlap=best_recall_iou,
                            score=float(face.score),
                            person_bbox=face.bbox,
                            frame_offset=0,
                        )
                        recall_ctx = _build_detection_context(face)
                        emit_sample(hid, None, assignment, recall_ctx, "recall")

            maybe_emit_status(frames_seen, processed_frames)
            for hid in list(sampling_states.keys()):
                selection_map.setdefault(hid, [])

        final_frames_seen = frame_idx + 1 if frame_idx >= 0 else 0
        frames_for_progress = frame_count or final_frames_seen
        emit_status(frames_for_progress, processed_frames, heartbeat=False)

        if guard_pending:
            for pending_hid, pending_state in list(guard_pending.items()):
                for sample_obj, embedding_vec, assign_obj, reason_label in zip(
                    pending_state.samples,
                    pending_state.embeddings,
                    pending_state.assignments,
                    pending_state.reasons,
                ):
                    _commit_sample(pending_hid, sample_obj, embedding_vec, assign_obj, reason_label)
            guard_pending.clear()

        finished_tracks: List[TrackState] = accumulator.flush()
        track_by_id: Dict[int, TrackState] = {t.track_id: t for t in finished_tracks}

        track_infos = _build_track_infos()
        stitched_tracks = _stitch_track_infos(track_infos)
        LOGGER.info(
            "Identity stitching evaluated %d tracks -> %d stitched tracks",
            len(track_infos),
            len(stitched_tracks),
        )

        old_to_new: Dict[int, int] = {}
        for stitched in stitched_tracks:
            for source_id in stitched.source_ids:
                old_to_new[source_id] = stitched.super_id

        stitched_byte_map: Dict[int, Optional[int]] = {}
        for stitched in stitched_tracks:
            unique_bytes = {byte_id for byte_id in stitched.byte_ids if byte_id is not None}
            if len(unique_bytes) == 1:
                stitched_byte_map[stitched.super_id] = next(iter(unique_bytes))
            else:
                stitched_byte_map[stitched.super_id] = None

        harvest_source_byte = stitched_byte_map
        selection_map = {track.super_id: track.samples for track in stitched_tracks}
        byte_to_harvest = {byte_id: old_to_new.get(hid, hid) for byte_id, hid in byte_to_harvest.items()}

        for track in stitched_tracks:
            for sample in track.samples:
                sample.track_id = track.super_id

        if candidate_rows:
            for row in candidate_rows:
                mapped = old_to_new.get(row.get("track_id"))
                if mapped is not None:
                    row["track_id"] = mapped

        for events in frame_events.values():
            for event in events:
                tid = event.get("track_id")
                if isinstance(tid, int) and tid in old_to_new:
                    event["track_id"] = old_to_new[tid]

        for events in split_events.values():
            for event in events:
                tid = event.get("track_id")
                if isinstance(tid, int) and tid in old_to_new:
                    event["track_id"] = old_to_new[tid]
                prev_id = event.get("prev_harvest_id")
                if isinstance(prev_id, int) and prev_id in old_to_new:
                    event["prev_harvest_id"] = old_to_new[prev_id]

        for hid, state in sampling_states.items():
            candidate_rows.extend(state.iter_debug_rows(hid))

        # Persist selected samples and compute track-level similarities
        total_selected = 0
        stitched_lookup: Dict[int, StitchedTrack] = {track.super_id: track for track in stitched_tracks}
        for stitched in stitched_tracks:
            samples = stitched.samples
            if not samples:
                continue
            if self.config.min_track_frames > 0 and stitched.total_frames < self.config.min_track_frames:
                LOGGER.debug(
                    "Skipping track %s due to min_track_frames=%s (duration=%s)",
                    stitched.super_id,
                    self.config.min_track_frames,
                    stitched.total_frames,
                )
                continue

            byte_id = harvest_source_byte.get(stitched.super_id)
            if not self.config.reindex_harvest_tracks and byte_id is not None:
                harvest_dirname = f"track_{byte_id:04d}"
            else:
                harvest_dirname = f"track_{stitched.super_id:04d}"
            track_dir = ensure_dir(crops_dir / harvest_dirname)

            embeddings: List[np.ndarray] = []
            for sample in samples:
                sample_filename = sample.path.name
                sample.path = track_dir / sample_filename
                if sample.embedding is None and self.embedder is not None and sample.image is not None and not self.config.defer_embeddings:
                    try:
                        sample.embedding = self.embedder.embed(sample.image)
                    except Exception as exc:  # pragma: no cover - defensive
                        LOGGER.warning(
                            "Post-filter embedding failed for track %s frame %s: %s",
                            stitched.super_id,
                            sample.frame_idx,
                            exc,
                        )
                        sample.embedding = None
                if sample.embedding is not None:
                    embeddings.append(sample.embedding)

            centroid: Optional[np.ndarray] = None
            if embeddings:
                centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
                norm = float(np.linalg.norm(centroid)) + 1e-9
                centroid = centroid / norm

            for sample in samples:
                if centroid is not None and sample.embedding is not None:
                    sample.similarity_to_centroid = _cosine(sample.embedding, centroid)
                target_dir = ensure_dir(sample.path.parent)
                if sample.image is not None:
                    cv2.imwrite(str(sample.path), sample.image)
                    sample.image = None
                abs_path = sample.path if sample.path.is_absolute() else (target_dir / sample.path.name).resolve()
                try:
                    rel_path = abs_path.relative_to(harvest_dir)
                except ValueError:
                    rel_path = Path(os.path.relpath(abs_path, harvest_dir))
                sample.path = rel_path

            total_selected += len(samples)

        # Flush queued debug writes (rejections, non-selected)
        if self.config.debug_rejections and debug_write_queue:
            for image, path in debug_write_queue:
                ensure_dir(path.parent)
                cv2.imwrite(str(path), image)
            debug_write_queue.clear()

        # Build manifest per harvest id, carrying source ByteTrack id
        manifest_entries: List[ManifestEntry] = []
        for stitched in stitched_tracks:
            samples = stitched.samples
            if not samples:
                continue
            byte_id = harvest_source_byte.get(stitched.super_id)
            entry = ManifestEntry(
                track_id=stitched.super_id,
                byte_track_id=byte_id,
                total_frames=stitched.total_frames,
                avg_conf=stitched.avg_conf,
                avg_area=stitched.avg_area,
                first_ts_ms=stitched.start_ts_ms,
                last_ts_ms=stitched.end_ts_ms,
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
                "identity_cosine",
                "similarity_to_centroid",
                "provider",
            ]
            with selected_csv.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                normalized_rows = []
                for row in candidate_rows:
                    row_copy = dict(row)
                    path_value = row_copy.get("path")
                    if path_value:
                        row_copy["path"] = str(Path(path_value))
                    else:
                        row_copy["path"] = ""
                    normalized_rows.append(row_copy)
                writer.writerows(normalized_rows)
            LOGGER.info("Harvest selection summary written to %s", selected_csv)

        if self.config.debug_rejections:
            debug_path = harvest_dir / "harvest_debug.json"
            payload: Dict[str, Any] = {
                "reject_counts": dict(reject_counter),
                "frame_events": {str(frame): events for frame, events in frame_events.items()},
            }
            if debug_records:
                payload["reject_details"] = {k: v for k, v in debug_records.items()}
            if split_events:
                payload["identity_splits"] = {
                    str(byte_id): events for byte_id, events in split_events.items()
                }
            dump_json(debug_path, payload)
            LOGGER.info("Harvest debug log written to %s", debug_path)

        LOGGER.info(
            "Harvest complete tracks=%d samples=%d manifest=%s",
            len(manifest_entries),
            total_selected,
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

    def _passes_face_size(self, bbox) -> bool:
        min_px = max(0, int(self.config.min_face_px))
        if min_px <= 0:
            return True
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return min(width, height) >= float(min_px)

    def _passes_sharpness_threshold(self, sharpness: float) -> bool:
        if self.config.min_sharpness_laplacian is None:
            return True
        return sharpness >= self.config.min_sharpness_laplacian

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
