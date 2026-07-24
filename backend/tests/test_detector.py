import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from detector import detect_and_save, render


class TestRender:
    def test_no_detections_returns_copy(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = render(img, [])
        assert result.shape == img.shape
        assert np.array_equal(result, img)

    def test_draws_bounding_boxes(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [{"bbox": [10, 10, 50, 50], "label": "car", "score": 0.95}]
        result = render(img, dets)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)

    def test_multiple_detections(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [
            {"bbox": [10, 10, 50, 50], "label": "car", "score": 0.95},
            {"bbox": [100, 100, 150, 150], "label": "person", "score": 0.85},
        ]
        result = render(img, dets)
        assert not np.array_equal(result, img)
        assert result.shape == img.shape

    def test_does_not_mutate_input(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        original = img.copy()
        dets = [{"bbox": [10, 10, 50, 50], "label": "car", "score": 0.95}]
        render(img, dets)
        assert np.array_equal(img, original)

    def test_zero_score_detection(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [{"bbox": [0, 0, 20, 20], "label": "unknown", "score": 0.0}]
        result = render(img, dets)
        assert result.shape == img.shape
        assert not np.array_equal(result, img)


class TestDetectAndSave:
    def test_nonexistent_image_returns_empty(self):
        result = detect_and_save("nonexistent_video.mp4", "car")
        assert result == []
