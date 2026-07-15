import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline import (
    frames_to_segments, parse_query, build_faiss_index, embed_text, _motion_score_diff
)

def test_frames_to_segments_merges_consecutive():
    indices = [0, 2, 4, 8, 10]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    segs = frames_to_segments(indices, scores, gap_thresh=3)
    assert len(segs) == 2, f"Expected 2 segments, got {len(segs)}"
    assert segs[0]["frame_indices"] == [0, 2, 4]
    assert segs[1]["frame_indices"] == [8, 10]

def test_frames_to_segments_empty():
    assert frames_to_segments([], []) == []

def test_frames_to_segments_single():
    segs = frames_to_segments([42], [0.95])
    assert len(segs) == 1
    assert segs[0]["frame_indices"] == [42]

def test_frames_to_segments_large_gap():
    indices = [0, 100]
    scores = [0.9, 0.8]
    segs = frames_to_segments(indices, scores, gap_thresh=3)
    assert len(segs) == 2

def test_parse_query_no_time_range():
    result = parse_query("person walking")
    assert result["semantic_query"] == "person walking"
    assert result["time_range"] is None

def test_parse_query_with_time_range():
    result = parse_query("find person between 12:30 and 13:45")
    assert result["time_range"] is not None
    assert "12:30" in result["time_range"]

def test_motion_score_diff():
    prev = np.zeros((90, 160), dtype=np.uint8)
    curr = np.ones((90, 160), dtype=np.uint8) * 255
    score = _motion_score_diff(prev, curr)
    assert score > 200, f"Expected high diff score, got {score}"

def test_motion_score_diff_identical():
    prev = np.zeros((90, 160), dtype=np.uint8)
    score = _motion_score_diff(prev, prev)
    assert score == 0.0, f"Expected 0 for identical frames, got {score}"

def test_build_faiss_index_small():
    embs = np.random.rand(10, 512).astype("float32")
    idx = build_faiss_index(embs)
    assert idx.ntotal == 10

def test_build_faiss_index_large():
    embs = np.random.rand(100, 512).astype("float32")
    idx = build_faiss_index(embs)
    assert idx.ntotal == 100

def test_embed_text_output_shape():
    emb = embed_text(["test query"])
    assert emb.shape[0] == 1
    assert emb.shape[1] == 512

def test_embed_text_multiple():
    embs = embed_text(["first", "second"])
    assert embs.shape[0] == 2
