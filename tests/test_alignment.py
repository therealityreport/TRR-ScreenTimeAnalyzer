import numpy as np

from screentime.detectors.face_retina import RetinaFaceDetector


def test_align_to_112_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (120.0, 160.0, 400.0, 440.0)
    aligned = RetinaFaceDetector.align_to_112(image, None, bbox)
    assert aligned.shape == (112, 112, 3)
