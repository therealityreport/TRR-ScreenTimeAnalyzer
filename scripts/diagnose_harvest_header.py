#!/usr/bin/env python3
"""
Diagnostic script to visualize harvest face detection and rejection reasons.

Usage:
    python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --frames 100-150
    python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --sample 50
"""

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.person_yolo import YOLOPersonDetector
from screentime.io_utils import load_yaml, setup_logging
from screentime.tracking.bytetrack_wrap import ByteTrackWrapper, TrackAccumulator
from screentime.types import bbox_area, iou

LOGGER = logging.getLogger("scripts.diagnose_harvest")
