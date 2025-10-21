from argparse import Namespace

import pytest

pytest.importorskip("cv2")

from scripts.harvest_faces import _resolve_min_frontalness


def test_resolve_min_frontalness_prefers_pipeline_config():
    pipeline_cfg = {"min_frontalness": 0.10}
    args = Namespace(frontalness_thresh=None, min_frontalness=None)

    resolved = _resolve_min_frontalness(args, pipeline_cfg)

    assert resolved == 0.10


def test_resolve_min_frontalness_prefers_cli_over_config():
    pipeline_cfg = {"min_frontalness": 0.10}
    args = Namespace(frontalness_thresh=0.25, min_frontalness=None)

    resolved = _resolve_min_frontalness(args, pipeline_cfg)

    assert resolved == 0.25


def test_resolve_min_frontalness_uses_legacy_default_when_missing():
    pipeline_cfg = {}
    args = Namespace(frontalness_thresh=None, min_frontalness=None)

    resolved = _resolve_min_frontalness(args, pipeline_cfg)

    assert resolved == 0.20
