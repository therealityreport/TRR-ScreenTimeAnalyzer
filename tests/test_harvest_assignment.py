from screentime.harvest.harvest import HarvestConfig, HarvestRunner
from screentime.tracking.bytetrack_wrap import TrackAccumulator
from screentime.types import TrackState


class DummyDetector:
    def __init__(self):
        self.outputs = []

    def detect(self, frame, frame_idx):  # pragma: no cover - stub
        return self.outputs.pop(0) if self.outputs else []


class DummyTracker:
    def __init__(self):
        self.frame_rate = None

    def set_frame_rate(self, fps):  # pragma: no cover
        self.frame_rate = fps


def build_runner():
    config = HarvestConfig()
    runner = HarvestRunner(DummyDetector(), DummyDetector(), DummyTracker(), config)
    return runner


def test_best_iou_track_picks_highest():
    runner = build_runner()
    track_lookup = {
        1: type("Obs", (), {"bbox": (0, 0, 80, 80)}),
        2: type("Obs", (), {"bbox": (40, 40, 160, 160)}),
    }
    face_bbox = (90, 90, 120, 120)
    accumulator = TrackAccumulator()
    state1 = TrackState(track_id=1, frames=[0], bboxes=[(0, 0, 80, 80)])
    state2 = TrackState(track_id=2, frames=[0], bboxes=[(40, 40, 160, 160)])
    accumulator.active[1] = state1
    accumulator.active[2] = state2
    track_id, iou_val, best_bbox = runner._best_iou_track(
        face_bbox,
        track_lookup,
        accumulator,
        frame_idx=0,
        frame_width=1920,
        frame_height=1080,
    )
    assert track_id == 2
    assert iou_val > 0.03
    assert best_bbox == (40, 40, 160, 160)


def test_face_center_in_bbox():
    runner = build_runner()
    person_bbox = (0, 0, 100, 200)
    face_inside_head = (10, 20, 30, 60)
    face_outside_head = (10, 120, 30, 160)
    assert runner._face_center_in_bbox(face_inside_head, person_bbox)
    outside = (150, 150, 200, 200)
    assert not runner._face_center_in_bbox(outside, person_bbox)


def test_quality_scores_frontal_higher():
    runner = build_runner()
    frontal = runner._score_quality(300.0, 1.0, runner.config.target_area_frac)
    profile = runner._score_quality(300.0, 0.2, runner.config.target_area_frac)
    assert frontal > profile


def test_default_face_in_track_iou_threshold():
    config = HarvestConfig()
    assert config.face_in_track_iou == 0.25


def test_center_fallback_requires_unique_candidate():
    runner = build_runner()
    faces = [
        (10, 10, 30, 30),
        (40, 40, 60, 60),
    ]
    person_bbox = (0, 0, 80, 80)
    candidate_faces = [idx for idx, bbox in enumerate(faces) if runner._face_center_in_bbox(bbox, person_bbox)]
    assert len(candidate_faces) == 2
    assert len(candidate_faces) != 1
