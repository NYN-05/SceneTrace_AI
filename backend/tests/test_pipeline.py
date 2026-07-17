import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline import (
    frames_to_segments, parse_query, build_faiss_index, embed_text, _motion_score_diff,
    SimpleTracker, MetadataDB
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


def test_tracker_empty():
    trk = SimpleTracker()
    assert trk.update([], 0) == []


def test_tracker_new_track():
    trk = SimpleTracker()
    dets = [{"bbox": [0, 0, 10, 10], "label": "car", "score": 0.9}]
    r = trk.update(dets, 0)
    assert len(r) == 1
    assert r[0]["track_id"] == 0
    assert r[0]["label"] == "car"


def test_tracker_iou_match():
    trk = SimpleTracker(match_thresh=0.3)
    trk.update([{"bbox": [0, 0, 10, 10], "label": "car", "score": 0.9}], 0)
    r = trk.update([{"bbox": [1, 1, 11, 11], "label": "car", "score": 0.85}], 1)
    assert len(r) == 1
    assert r[0]["track_id"] == 0


def test_tracker_new_track_on_low_iou():
    trk = SimpleTracker(match_thresh=0.8)
    trk.update([{"bbox": [0, 0, 10, 10], "label": "car", "score": 0.9}], 0)
    r = trk.update([{"bbox": [50, 50, 60, 60], "label": "car", "score": 0.85}], 1)
    assert len(r) == 1
    assert r[0]["track_id"] == 1


def test_tracker_multiple_classes():
    trk = SimpleTracker(match_thresh=0.3)
    trk.update([{"bbox": [0, 0, 10, 10], "label": "car", "score": 0.9}], 0)
    r = trk.update([{"bbox": [1, 1, 11, 11], "label": "person", "score": 0.85}], 1)
    assert len(r) == 1
    assert r[0]["track_id"] == 1


def test_tracker_summary():
    trk = SimpleTracker(match_thresh=0.3)
    trk.update([{"bbox": [0, 0, 10, 10], "label": "car", "score": 0.9}], 0)
    trk.update([{"bbox": [2, 2, 12, 12], "label": "car", "score": 0.85}], 1)
    trk.update([{"bbox": [4, 4, 14, 14], "label": "car", "score": 0.8}], 2)
    s = trk.summary()
    assert "0" in s
    assert s["0"]["total_frames"] == 3
    assert s["0"]["class"] == "car"


def test_tracker_summary_skips_short():
    trk = SimpleTracker()
    trk.update([{"bbox": [0, 0, 10, 10], "label": "car", "score": 0.9}], 0)
    s = trk.summary()
    assert len(s) == 0


def test_metadata_db_create_and_populate(tmpdir):
    db_path = Path(tmpdir) / "test.db"
    mdb = MetadataDB(db_path)
    objs = [
        {"frame_idx": 0, "bbox": [0, 0, 10, 10], "label": "car", "score": 0.9, "track_id": 0},
        {"frame_idx": 1, "bbox": [5, 5, 15, 15], "label": "person", "score": 0.85, "track_id": 1},
    ]
    tracks = {"0": {"class": "car", "start_frame": 0, "end_frame": 5, "total_frames": 3,
                     "avg_confidence": 0.9, "displacement": 10.0}}
    mdb.populate(objs, tracks, [0, 1], [0.0, 0.033])
    mdb.close()
    assert db_path.exists()
    assert db_path.stat().st_size > 0


def test_metadata_db_query_objects_all(tmpdir):
    db_path = Path(tmpdir) / "test.db"
    mdb = MetadataDB(db_path)
    objs = [
        {"frame_idx": 0, "bbox": [0, 0, 10, 10], "label": "car", "score": 0.9, "track_id": 0},
        {"frame_idx": 1, "bbox": [5, 5, 15, 15], "label": "person", "score": 0.85, "track_id": 1},
    ]
    mdb.populate(objs, {}, [0, 1], [0.0, 0.033])
    results = mdb.query_objects()
    mdb.close()
    assert len(results) == 2


def test_metadata_db_query_by_class(tmpdir):
    db_path = Path(tmpdir) / "test.db"
    mdb = MetadataDB(db_path)
    objs = [
        {"frame_idx": 0, "bbox": [0, 0, 10, 10], "label": "car", "score": 0.9, "track_id": 0},
        {"frame_idx": 1, "bbox": [5, 5, 15, 15], "label": "person", "score": 0.85, "track_id": 1},
    ]
    mdb.populate(objs, {}, [0, 1], [0.0, 0.033])
    results = mdb.query_objects(class_name="car")
    mdb.close()
    assert len(results) == 1
    assert results[0]["class"] == "car"


def test_metadata_db_query_by_confidence(tmpdir):
    db_path = Path(tmpdir) / "test.db"
    mdb = MetadataDB(db_path)
    objs = [
        {"frame_idx": 0, "bbox": [0, 0, 10, 10], "label": "car", "score": 0.9, "track_id": 0},
        {"frame_idx": 1, "bbox": [5, 5, 15, 15], "label": "car", "score": 0.5, "track_id": 1},
    ]
    mdb.populate(objs, {}, [0, 1], [0.0, 0.033])
    results = mdb.query_objects(class_name="car", min_confidence=0.7)
    mdb.close()
    assert len(results) == 1
    assert results[0]["confidence"] == 0.9


def test_metadata_db_get_track(tmpdir):
    db_path = Path(tmpdir) / "test.db"
    mdb = MetadataDB(db_path)
    tracks = {"0": {"class": "car", "start_frame": 0, "end_frame": 5, "total_frames": 3,
                     "avg_confidence": 0.9, "displacement": 10.0}}
    mdb.populate([], tracks, [], [])
    trk = mdb.get_track(0)
    mdb.close()
    assert trk is not None
    assert trk["class"] == "car"
    assert trk["total_frames"] == 3


def test_metadata_db_get_track_missing(tmpdir):
    db_path = Path(tmpdir) / "test.db"
    mdb = MetadataDB(db_path)
    trk = mdb.get_track(999)
    mdb.close()
    assert trk is None

def test_embed_text_output_shape():
    emb = embed_text(["test query"])
    assert emb.shape[0] == 1
    assert emb.shape[1] == 512

def test_embed_text_multiple():
    embs = embed_text(["first", "second"])
    assert embs.shape[0] == 2
