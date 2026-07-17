import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
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
    assert stats["indexing_speed_fps"] == pytest.approx(285.7, 0.1)

def test_query_cap():
    bm = BenchmarkStats()
    bm._reset()
    for _ in range(250):
        bm.record_query(0.1)
    stats = bm.stats()
    assert stats["total_queries"] == 200
