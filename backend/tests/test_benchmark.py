import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BenchmarkStats


def test_singleton():
    a = BenchmarkStats()
    b = BenchmarkStats()
    assert a is b


def test_record_query():
    bm = BenchmarkStats()
    bm._reset()
    bm.record_query(0.5)
    bm.record_query(1.0)
    stats = bm.stats()
    assert stats["total_queries"] == 2
    assert stats["avg_query_latency"] == 0.75


def test_record_index():
    bm = BenchmarkStats()
    bm._reset()
    bm.record_index(10, 5, 20, 35, 10000, 500)
    stats = bm.stats()
    assert stats["videos_indexed"] == 1
    assert stats["total_frames"] == 10000
    assert stats["total_keyframes"] == 500
    assert stats["total_index_time"] == 35.0
    assert stats["indexing_speed_fps"] == round(10000 / 35, 1)
    assert stats["avg_frame_reduction_pct"] > 0


def test_query_cap():
    bm = BenchmarkStats()
    bm._reset()
    for _ in range(250):
        bm.record_query(0.1)
    stats = bm.stats()
    assert stats["total_queries"] == 200


def test_uptime():
    bm = BenchmarkStats()
    bm._reset()
    stats = bm.stats()
    assert stats["uptime_seconds"] >= 0


def test_multiple_index_records():
    bm = BenchmarkStats()
    bm._reset()
    bm.record_index(10, 5, 20, 35, 10000, 500)
    bm.record_index(15, 8, 30, 53, 20000, 1000)
    stats = bm.stats()
    assert stats["videos_indexed"] == 2
    assert stats["total_frames"] == 30000
    assert stats["avg_scan_time"] == 12.5
    assert stats["avg_extract_time"] == 6.5
