"""Cosine similarity matcher with vote persistence for tracks."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from screentime.types import TrackState, l2_normalize

LOGGER = logging.getLogger("screentime.recognition.matcher")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Embedding shapes do not match")
    return float(np.dot(a, b))


class TrackVotingMatcher:
    """Maintains per-track vote state given incoming embeddings."""

    def __init__(
        self,
        facebank: Dict[str, np.ndarray],
        similarity_th: float = 0.80,
        vote_decay: float = 0.99,
        flip_tolerance: float = 0.30,
        dominance_ratio: float = 2.0,
        identity_split_enabled: bool = True,
        identity_split_min_frames: int = 5,
        identity_change_margin: float = 0.08,
        per_label_th: Optional[Dict[str, float]] = None,
        min_margin: float = 0.0,
    ) -> None:
        self.facebank = {label: l2_normalize(vec) for label, vec in facebank.items()}
        self.similarity_th = similarity_th
        self.vote_decay = vote_decay
        self.flip_tolerance = flip_tolerance
        self.dominance_ratio = dominance_ratio
        self.identity_split_enabled = identity_split_enabled
        self.identity_split_min_frames = identity_split_min_frames
        self.identity_change_margin = identity_change_margin
        self.per_label_th = per_label_th or {}
        self.min_margin = min_margin

    def best_match(self, embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        if not self.facebank:
            return None
        embedding = l2_normalize(embedding)
        scores = [
            (label, cosine_similarity(vec, embedding))
            for label, vec in self.facebank.items()
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores:
            return None
        top_label, top_score = scores[0]
        runner_score = scores[1][1] if len(scores) > 1 else float("-inf")
        required = self.per_label_th.get(top_label, self.similarity_th)
        if top_score < required:
            return None
        if (top_score - runner_score) < self.min_margin:
            return None
        return top_label, top_score

    def topk(self, embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Return the top-k matches without applying the similarity threshold."""
        if not self.facebank:
            return []
        embedding = l2_normalize(embedding)
        sims = [
            (label, float(np.dot(vec, embedding)))
            for label, vec in self.facebank.items()
        ]
        sims.sort(key=lambda item: item[1], reverse=True)
        return sims[:k]

    def update_track(self, track: TrackState, embedding: np.ndarray, frame_idx: Optional[int] = None) -> None:
        match = self.best_match(embedding)
        # Decay existing votes
        for label in list(track.label_votes.keys()):
            track.label_votes[label] *= self.vote_decay

        if match is None:
            # No match - reset change candidate
            if self.identity_split_enabled:
                track.label_change_candidate = None
                track.label_change_streak = 0
            return

        label, score = match
        track.record_vote(label, score)
        
        # Store face match for this frame
        if frame_idx is not None:
            track.face_matches[frame_idx] = (label, score)

        # Identity split logic: check if we need to end current subtrack
        if self.identity_split_enabled:
            self._check_identity_split(track, label, score, frame_idx)

        sorted_votes = sorted(track.label_votes.items(), key=lambda item: item[1], reverse=True)
        if not sorted_votes:
            return

        top_label, top_weight = sorted_votes[0]
        runner_up_weight = sorted_votes[1][1] if len(sorted_votes) > 1 else 0.0

        # Check if dominance condition holds to set/keep identity
        if top_weight >= runner_up_weight * self.dominance_ratio:
            previous_label = track.label
            if previous_label is None:
                track.label = top_label
                if self.identity_split_enabled and track.current_subtrack_label is None:
                    track.current_subtrack_label = top_label
                return
            if previous_label == top_label:
                # reinforce the label
                return

            # Evaluate flip tolerance
            current_weight = track.label_votes.get(previous_label, 0.0)
            if top_weight >= current_weight * (1.0 + self.flip_tolerance):
                LOGGER.debug(
                    "Track %s flipping label %s -> %s (%.3f vs %.3f)",
                    track.track_id,
                    previous_label,
                    top_label,
                    top_weight,
                    current_weight,
                )
                track.label = top_label
        else:
            LOGGER.debug(
                "Track %s votes not dominant: %s",
                track.track_id,
                sorted_votes,
            )

    def _check_identity_split(self, track: TrackState, new_label: str, similarity: float, frame_idx: Optional[int] = None) -> None:
        """Check if identity changed and split subtrack if confirmed."""
        current_label = track.current_subtrack_label

        if current_label is None:
            # First label for this track
            track.current_subtrack_label = new_label
            track.current_subtrack_start_frame = frame_idx
            track.label_change_candidate = None
            track.label_change_streak = 0
            LOGGER.debug(
                "Track %s start subtrack=%s at frame=%s",
                track.track_id,
                new_label,
                frame_idx,
            )
            return

        if new_label == current_label:
            # Same label - reset candidate
            track.label_change_candidate = None
            track.label_change_streak = 0
            return

        # Different label detected
        if track.label_change_candidate == new_label:
            track.label_change_streak += 1
        else:
            track.label_change_candidate = new_label
            track.label_change_streak = 1

        # Check if we should split
        if track.label_change_streak >= self.identity_split_min_frames:
            # Get similarity scores for both labels
            current_score = track.label_votes.get(current_label, 0.0)
            new_score = track.label_votes.get(new_label, 0.0)

            # Only split if new label has sufficient margin over current
            if new_score >= current_score + self.identity_change_margin:
                split_frame = frame_idx if frame_idx is not None else (track.frames[-1] if track.frames else -1)
                LOGGER.info(
                    "Track %s identity split: %s -> %s at frame %d (%.3f vs %.3f, streak=%d)",
                    track.track_id,
                    current_label,
                    new_label,
                    split_frame,
                    new_score,
                    current_score,
                    track.label_change_streak,
                )
                self._finalize_subtrack(track, current_label, split_frame)
                track.current_subtrack_label = new_label
                track.current_subtrack_start_frame = split_frame
                track.label_change_candidate = None
                track.label_change_streak = 0

    def _finalize_subtrack(self, track: TrackState, label: str, end_frame: Optional[int] = None) -> None:
        """Create a subtrack from current segment and add to track."""
        from screentime.types import Subtrack

        start_frame = track.current_subtrack_start_frame
        if start_frame is None:
            start_frame = track.frames[0] if track.frames else None
        if start_frame is None:
            return

        if end_frame is not None:
            window_end = end_frame - 1
        else:
            window_end = track.frames[-1] if track.frames else start_frame

        if window_end < start_frame:
            window_end = start_frame

        frame_scores: Dict[int, float] = {}
        for frame_idx in sorted(track.label_scores.keys()):
            if frame_idx < start_frame:
                continue
            if frame_idx > window_end:
                break
            assigned_label, similarity = track.label_scores[frame_idx]
            if assigned_label == label:
                frame_scores[frame_idx] = similarity

        if not frame_scores:
            LOGGER.warning(
                "Track %d: subtrack %s has no frame_scores in frames %d-%d; avg similarity will be NaN",
                track.track_id,
                label,
                start_frame,
                window_end,
            )

        subtrack = Subtrack(
            start_frame=start_frame,
            end_frame=window_end,
            label=label,
            frame_scores=frame_scores,
        )
        track.subtracks.append(subtrack)
        subtrack_idx = len(track.subtracks) - 1
        track.current_subtrack_start_frame = None
        LOGGER.info(
            "Finalized subtrack %d.%d: %s (window %d-%d, %d labeled frames, avg_sim=%s)",
            track.track_id,
            subtrack_idx,
            label,
            subtrack.start_frame,
            subtrack.end_frame,
            len(subtrack.frame_scores),
            f"{subtrack.avg_similarity:.3f}" if subtrack.frame_scores else "nan",
        )
