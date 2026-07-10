"""Thread-safe performance benchmark tracking for the dashboard.
Records indexing speed, query latency, and GPU metrics."""
import time
import threading

class BenchmarkStats:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._reset()
        return cls._instance

    def _reset(self):
        self.videos_indexed = 0
        self.total_frames = 0
        self.total_keyframes = 0
        self.total_scan_time = 0.0
        self.total_extract_time = 0.0
        self.total_embed_time = 0.0
        self.total_index_time = 0.0
        self._query_times = []
        self._start = time.time()

    def record_index(self, scan_t, extract_t, embed_t, total_t, total_frames, keyframes):
        with self._lock:
            self.videos_indexed += 1
            self.total_frames += total_frames
            self.total_keyframes += keyframes
            self.total_scan_time += scan_t
            self.total_extract_time += extract_t
            self.total_embed_time += embed_t
            self.total_index_time += total_t

    def record_query(self, seconds: float):
        with self._lock:
            self._query_times.append(seconds)
            if len(self._query_times) > 200:
                self._query_times = self._query_times[-200:]

    def stats(self):
        with self._lock:
            uptime = time.time() - self._start
            q = self._query_times
            avg_q = sum(q) / len(q) if q else 0
            reduction = 0
            if self.total_frames > 0:
                reduction = round((1 - self.total_keyframes / self.total_frames) * 100, 1)
            speed = 0
            if self.total_index_time > 0:
                speed = round(self.total_frames / self.total_index_time, 1)
            return {
                "uptime_seconds": round(uptime),
                "videos_indexed": self.videos_indexed,
                "total_frames": self.total_frames,
                "total_keyframes": self.total_keyframes,
                "avg_frame_reduction_pct": reduction,
                "total_index_time": round(self.total_index_time, 1),
                "avg_scan_time": round(self.total_scan_time / max(self.videos_indexed, 1), 1),
                "avg_extract_time": round(self.total_extract_time / max(self.videos_indexed, 1), 1),
                "avg_embed_time": round(self.total_embed_time / max(self.videos_indexed, 1), 1),
                "avg_index_time": round(self.total_index_time / max(self.videos_indexed, 1), 1),
                "indexing_speed_fps": speed,
                "avg_query_latency": round(avg_q, 3),
                "total_queries": len(q),
            }

benchmark = BenchmarkStats()
