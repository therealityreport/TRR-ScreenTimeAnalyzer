"""Scene detection helpers."""

from __future__ import annotations

from typing import List, Tuple

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_len: int = 12) -> List[Tuple[int, int]]:
    """Return a list of inclusive frame ranges [(start_frame, end_frame)]."""
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video)

    ranges: List[Tuple[int, int]] = []
    for scene_start, scene_end in sm.get_scene_list():
        start_frame = scene_start.get_frames()
        end_frame = scene_end.get_frames() - 1
        if end_frame >= start_frame:
            ranges.append((start_frame, end_frame))
    return ranges
