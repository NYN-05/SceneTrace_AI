import cv2
import numpy as np
import torch
from pathlib import Path
import faiss
import json
import re
import logging
import threading
import sqlite3
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import benchmark as bm
from config import settings, logger
from detector import detect

STORAGE = Path(__file__).parent / "storage"
for _d in ["originals", "frames", "clips", "reports"]:
    (STORAGE / _d).mkdir(parents=True, exist_ok=True)

device = settings.DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

_faiss_index_cache: dict[str, faiss.Index] = {}
_faiss_cache_lock = threading.Lock()
_HAS_FAISS_INDEX = set()

@dataclass
class VideoIndex:
    video_id: str
    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    motion_scores: list[float] = field(default_factory=list)
    total_frames: int = 0
    metadata: dict = field(default_factory=lambda: {"fps": 0, "width": 0, "height": 0, "duration": 0})
    benchmarks: dict = field(default_factory=lambda: {"scan_time": 0, "extract_time": 0, "embed_time": 0, "total_time": 0})
    object_metadata: list[dict] = field(default_factory=list)
    track_metadata: dict = field(default_factory=dict)
    captions: dict[int, str] = field(default_factory=dict)
    clip_indices: list[list[int]] = field(default_factory=list)

    def faiss_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "index.faiss"

    def embeddings_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "embeddings.npy"

    def get_faiss_index(self) -> faiss.Index | None:
        with _faiss_cache_lock:
            if self.video_id in _faiss_index_cache:
                return _faiss_index_cache[self.video_id]
        p = self.faiss_path()
        if p.exists():
            idx = faiss.read_index(str(p))
            with _faiss_cache_lock:
                _faiss_index_cache[self.video_id] = idx
            return idx
        return None

    def save_faiss_index(self, index: faiss.Index):
        p = self.faiss_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(p))
        with _faiss_cache_lock:
            _faiss_index_cache[self.video_id] = index
        _HAS_FAISS_INDEX.add(self.video_id)

    def invalidate_faiss_cache(self):
        with _faiss_cache_lock:
            _faiss_index_cache.pop(self.video_id, None)

    def load_embeddings(self) -> np.ndarray | None:
        p = self.embeddings_path()
        if p.exists():
            return np.load(str(p)).astype("float32")
        return None

    def save_embeddings(self, embeddings: np.ndarray):
        p = self.embeddings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), embeddings)

    def object_embeddings_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "object_embeddings.npy"

    def object_faiss_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "object_index.faiss"

    def get_object_faiss_index(self) -> faiss.Index | None:
        cache_key = f"{self.video_id}_objects"
        with _faiss_cache_lock:
            if cache_key in _faiss_index_cache:
                return _faiss_index_cache[cache_key]
        p = self.object_faiss_path()
        if p.exists():
            idx = faiss.read_index(str(p))
            with _faiss_cache_lock:
                _faiss_index_cache[cache_key] = idx
            return idx
        return None

    def save_object_faiss_index(self, index: faiss.Index):
        p = self.object_faiss_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(p))
        with _faiss_cache_lock:
            _faiss_index_cache[f"{self.video_id}_objects"] = index

    def load_object_embeddings(self) -> np.ndarray | None:
        p = self.object_embeddings_path()
        if p.exists():
            return np.load(str(p)).astype("float32")
        return None

    def save_object_embeddings(self, embeddings: np.ndarray):
        p = self.object_embeddings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), embeddings)

    def caption_embeddings_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "caption_embeddings.npy"

    def caption_faiss_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "caption_index.faiss"

    def get_caption_faiss_index(self) -> faiss.Index | None:
        cache_key = f"{self.video_id}_captions"
        with _faiss_cache_lock:
            if cache_key in _faiss_index_cache:
                return _faiss_index_cache[cache_key]
        p = self.caption_faiss_path()
        if p.exists():
            idx = faiss.read_index(str(p))
            with _faiss_cache_lock:
                _faiss_index_cache[cache_key] = idx
            return idx
        return None

    def save_caption_faiss_index(self, index: faiss.Index):
        p = self.caption_faiss_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(p))
        with _faiss_cache_lock:
            _faiss_index_cache[f"{self.video_id}_captions"] = index

    def load_caption_embeddings(self) -> np.ndarray | None:
        p = self.caption_embeddings_path()
        if p.exists():
            return np.load(str(p)).astype("float32")
        return None

    def save_caption_embeddings(self, embeddings: np.ndarray):
        p = self.caption_embeddings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), embeddings)

    def clip_embeddings_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "clip_embeddings.npy"

    def load_clip_embeddings(self) -> np.ndarray | None:
        p = self.clip_embeddings_path()
        if p.exists():
            return np.load(str(p)).astype("float32")
        return None

    def save_clip_embeddings(self, embeddings: np.ndarray):
        p = self.clip_embeddings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), embeddings)

    def metadata_db_path(self) -> Path:
        return STORAGE / "frames" / self.video_id / "metadata.db"

from transformers import CLIPModel, CLIPProcessor
_clip_model = None
_clip_processor = None
_clip_lock = threading.Lock()

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        with _clip_lock:
            if _clip_model is None:
                logger.info("Loading CLIP model on %s...", device)
                _clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).eval().to(device)
                _clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
                logger.info("CLIP model loaded")
    return _clip_model, _clip_processor

def _preprocess_batch(batch: list[np.ndarray], processor) -> dict:
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
    return processor(images=rgb, return_tensors="pt", padding=True)

@torch.inference_mode()
def compute_embeddings(imgs: list[np.ndarray], batch_size: int = 0,
                       progress: dict = None) -> np.ndarray:
    model, processor = _get_clip()
    all_embs = []
    bs = batch_size or settings.BATCH_SIZE
    n_batches = max(1, (len(imgs) + bs - 1) // bs)
    pool = ThreadPoolExecutor(max_workers=1)
    preproc_future = None
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i+bs]
        if preproc_future is not None:
            inputs = preproc_future.result().to(device)
        else:
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
            inputs = processor(images=rgb, return_tensors="pt", padding=True).to(device)
        next_i = i + bs
        if next_i < len(imgs):
            next_batch = imgs[next_i:next_i+bs]
            preproc_future = pool.submit(_preprocess_batch, next_batch, processor)
        else:
            preproc_future = None
        emb = model.get_image_features(**inputs)
        emb_norm = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb_norm.cpu().numpy())
        if progress is not None:
            batch_idx = i // bs + 1
            pct = min(99, 25 + int(60 * batch_idx / n_batches))
            progress["percent"] = pct
            progress["message"] = f"Embedding: batch {batch_idx}/{n_batches}"
            elapsed = time.time() - progress["_t0"]
            eff = max(pct, 1)
            progress["eta_seconds"] = int(elapsed / eff * (100 - eff))
    pool.shutdown(wait=False)
    return np.concatenate(all_embs, axis=0).astype("float32")

@torch.inference_mode()
def embed_text(texts: list[str]) -> np.ndarray:
    model, processor = _get_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")


@torch.inference_mode()
def embed_image(image: np.ndarray) -> np.ndarray:
    model, processor = _get_clip()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")[0]


# ── Phase 4: Clip-level embedding ─────────────────────────

def _encode_clip_light(frame_embeddings: np.ndarray, motion_scores: list[float]) -> np.ndarray:
    """Combine averaged frame embeddings + motion stats into a single clip vector.

    Uses already-computed frame embeddings (L2-normalized) and motion scores
    from motion sampling. No optical flow needed.
    """
    avg_emb = np.mean(frame_embeddings, axis=0)
    ms = np.array(motion_scores, dtype="float32")
    mw = settings.CLIP_MOTION_WEIGHT
    motion_feat = np.array([float(np.mean(ms)), float(np.std(ms)), float(np.max(ms))], dtype="float32")
    clip_vec = np.concatenate([avg_emb * (1 - mw), motion_feat * mw])
    norm = np.linalg.norm(clip_vec)
    if norm > 0:
        clip_vec = clip_vec / norm
    return clip_vec


# ── Phase 3: Optional scene captioner ──────────────────────

_CAPTIONER_LOCK = threading.Lock()
_CAPTIONER_INSTANCE = None

class Captioner:
    """Florence-2 caption generator. Lazy-loaded; graceful fallback."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.CAPTIONER_MODEL
        self._processor = None
        self._model = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        logger.info("Loading captioner '%s' on %s...", self.model_name, device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype="auto",
        ).eval().to(device)
        logger.info("Captioner loaded")

    def caption(self, image: np.ndarray) -> str:
        if self._model is None:
            try:
                self._load()
            except Exception:
                logger.exception("Captioner failed to load; disabling")
                self._model = False  # cache failure so we don't retry every frame
        if self._model is False:
            return ""
        with torch.inference_mode():
            prompt = "<MORE_DETAILED_CAPTURE>"
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = self._processor(text=prompt, images=rgb,
                                     return_tensors="pt").to(device)
            outputs = self._model.generate(
                **inputs, max_new_tokens=512, num_beams=3)
            return self._processor.decode(outputs[0], skip_special_tokens=True)


def _get_captioner() -> Captioner | None:
    global _CAPTIONER_INSTANCE
    if _CAPTIONER_INSTANCE is None:
        with _CAPTIONER_LOCK:
            if _CAPTIONER_INSTANCE is None:
                try:
                    _CAPTIONER_INSTANCE = Captioner()
                except Exception:
                    logger.exception("Failed to load captioner; captioning disabled")
                    _CAPTIONER_INSTANCE = False
    return _CAPTIONER_INSTANCE if _CAPTIONER_INSTANCE is not False else None


class SimpleTracker:
    """IoU-based multi-object tracker. No external dependencies."""
    def __init__(self, match_thresh: float = 0.5, track_buffer: int = 30):
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self._tracks: dict[int, dict] = {}
        self._next_id = 0
        self._frame_num = 0

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections: list[dict], frame_idx: int) -> list[dict]:
        self._frame_num += 1
        if not detections:
            for t in self._tracks.values():
                t["active"] = False
            return []

        assigned_dets: set[int] = set()
        results: list[dict] = []

        for tid in sorted(self._tracks):
            trk = self._tracks[tid]
            if not trk["active"]:
                continue
            best_iou, best_di = self.match_thresh, -1
            for di, det in enumerate(detections):
                if di in assigned_dets or det["label"] != trk["label"]:
                    continue
                iou = self._iou(det["bbox"], trk["last_bbox"])
                if iou > best_iou:
                    best_iou, best_di = iou, di
            if best_di >= 0:
                assigned_dets.add(best_di)
                d = detections[best_di]
                trk["frames"].append(frame_idx)
                trk["bboxes"].append(d["bbox"])
                trk["scores"].append(d["score"])
                trk["last_bbox"] = d["bbox"]
                trk["last_seen"] = self._frame_num
                results.append({**d, "track_id": tid})

        for di, d in enumerate(detections):
            if di in assigned_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "label": d["label"], "frames": [frame_idx],
                "bboxes": [d["bbox"]], "scores": [d["score"]],
                "last_bbox": d["bbox"], "last_seen": self._frame_num,
                "active": True,
            }
            results.append({**d, "track_id": tid})

        for t in self._tracks.values():
            if self._frame_num - t["last_seen"] > self.track_buffer:
                t["active"] = False
        return results

    def summary(self) -> dict:
        out = {}
        for tid, t in self._tracks.items():
            if len(t["frames"]) < 2:
                continue
            b = t["bboxes"]
            cx = [(b[i][0] + b[i][2]) / 2 for i in range(len(b))]
            cy = [(b[i][1] + b[i][3]) / 2 for i in range(len(b))]
            displacement = ((cx[-1] - cx[0])**2 + (cy[-1] - cy[0])**2) ** 0.5
            out[str(tid)] = {
                "class": t["label"], "start_frame": t["frames"][0],
                "end_frame": t["frames"][-1], "total_frames": len(t["frames"]),
                "avg_confidence": round(sum(t["scores"]) / len(t["scores"]), 4),
                "displacement": round(displacement, 2),
            }
        return out


class MetadataDB:
    """Per-video SQLite metadata store for queryable object/track lookup."""

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY,
                frame_idx INTEGER, track_id INTEGER DEFAULT -1,
                class TEXT, confidence REAL,
                bbox_x1 INTEGER, bbox_y1 INTEGER,
                bbox_x2 INTEGER, bbox_y2 INTEGER,
                timestamp REAL,
                motion_score REAL DEFAULT 0
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                class TEXT, start_frame INTEGER, end_frame INTEGER,
                total_frames INTEGER, avg_confidence REAL, displacement REAL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_class ON objects(class)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_obj_track ON objects(track_id)")
        self.conn.commit()

    def populate(self, object_metadata: list[dict], track_metadata: dict,
                 frame_indices: list[int], timestamps: list[float]):
        frame_to_ts = dict(zip(frame_indices, timestamps))
        objs_data = [
            (obj["frame_idx"], obj.get("track_id", -1), obj["label"], obj["score"],
             obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3],
             frame_to_ts.get(obj["frame_idx"], 0.0))
            for obj in object_metadata
        ]
        if objs_data:
            self.conn.executemany(
                "INSERT INTO objects (frame_idx, track_id, class, confidence, "
                "bbox_x1, bbox_y1, bbox_x2, bbox_y2, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", objs_data)
        tracks_data = [
            (int(tid_str), td["class"], td["start_frame"], td["end_frame"],
             td["total_frames"], td["avg_confidence"], td["displacement"])
            for tid_str, td in track_metadata.items()
        ]
        if tracks_data:
            self.conn.executemany(
                "INSERT OR REPLACE INTO tracks "
                "(track_id, class, start_frame, end_frame, total_frames, avg_confidence, displacement) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", tracks_data)
        self.conn.commit()

    def query_objects(self, class_name: str = None, track_id: int = None,
                      min_confidence: float = 0.0) -> list[dict]:
        clauses = ["confidence >= ?"]
        params: list = [min_confidence]
        if class_name:
            clauses.append("class = ?")
            params.append(class_name)
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        rows = self.conn.execute(
            "SELECT * FROM objects WHERE " + " AND ".join(clauses), params).fetchall()
        return [dict(r) for r in rows]

    def get_track(self, track_id: int) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM tracks WHERE track_id = ?", (track_id,)).fetchone()
        return dict(r) if r else None

    def close(self):
        self.conn.close()


def _motion_score_farneback(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())

def _motion_score_diff(prev_gray, gray):
    return float(np.abs(gray.astype("i2") - prev_gray.astype("i2")).mean())

def motion_sample(video_path: str, stride: int = 0, target_pct: float = 0,
                  progress: dict = None, method: str = "diff") -> tuple[list[int], list[float], list[float], int]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    score_fn = _motion_score_farneback if method == "farneback" else _motion_score_diff
    stride = stride or settings.MOTION_STRIDE
    target_pct = target_pct or settings.MOTION_TARGET_PCT
    start = time.time()
    candidate_frames, timestamps, motion_mags = [], [], []
    prev_gray = None
    count = 0
    report_every = max(1, total_frames // 20)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % report_every == 0 and progress is not None:
            pct = min(18, int(18 * count / max(total_frames, 1)))
            progress["percent"] = pct
            progress["message"] = f"Motion scan: {count}/{total_frames} frames"
            elapsed = time.time() - progress["_t0"]
            if pct > 0:
                progress["eta_seconds"] = int(elapsed / pct * (100 - pct))
        if count % stride != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            mag = score_fn(prev_gray, gray)
            motion_mags.append(mag)
            candidate_frames.append(count - 1)
            timestamps.append((count - 1) / fps)
        prev_gray = gray
    cap.release()
    elapsed = time.time() - start
    if not motion_mags:
        return [0], [0.0], [0.0], total_frames
    thresh = np.percentile(motion_mags, max(0, 100 - target_pct))
    kept = [(f, t, m) for f, t, m in zip(candidate_frames, timestamps, motion_mags) if m >= thresh]
    if not kept:
        kept = [(candidate_frames[0], timestamps[0], motion_mags[0])]
    keep_frames, ts, sc = zip(*kept)
    keep_frames = list(keep_frames)
    ts = list(ts)
    sc = list(sc)
    if 0 not in keep_frames:
        keep_frames.insert(0, 0)
        ts.insert(0, 0.0)
        sc.insert(0, sc[0] if sc else 0.0)
    logger.info("Motion scan: %d frames in %.1fs (%s), thresh=%.2f, kept=%d (%d%%)",
                count, elapsed, method, thresh, len(keep_frames), 100*len(keep_frames)//max(count,1))
    return keep_frames, ts, sc, total_frames

def _extract_chunk(video_path: str, indices: list[int]) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    result = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            result.append((idx, frame))
    cap.release()
    return result

def extract_frames_parallel(video_path: str, frame_indices: list[int], num_workers: int = 0) -> list[np.ndarray]:
    if not frame_indices:
        return []
    nw = num_workers or settings.NUM_WORKERS
    chunks = np.array_split(frame_indices, min(nw, len(frame_indices)))
    with ThreadPoolExecutor(max_workers=nw) as pool:
        futures = [pool.submit(_extract_chunk, video_path, list(chunk)) for chunk in chunks if len(chunk) > 0]
        all_frames = []
        for f in as_completed(futures):
            all_frames.extend(f.result())
    all_frames.sort(key=lambda x: x[0])
    return [frame for _, frame in all_frames]

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    n, d = embeddings.shape
    if n < settings.FAISS_IVF_THRESHOLD:
        idx = faiss.IndexFlatIP(d)
        idx.add(embeddings)
        return idx
    nlist = max(1, int(np.sqrt(n)))
    quant = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(embeddings)
    idx.add(embeddings)
    idx.nprobe = max(1, nlist // 4)
    return idx

def search_embeddings(query: str, index: faiss.Index | None, embeddings: np.ndarray, top_k: int = 10) -> tuple[list[int], list[float]]:
    q_emb = embed_text([query])
    n = len(embeddings)
    if n == 0:
        return [], []
    if index is None:
        index = build_faiss_index(embeddings)
    scores, indices = index.search(q_emb, min(top_k, n))
    return indices[0].tolist(), scores[0].tolist()

def frames_to_segments(indices: list[int], scores: list[float], gap_thresh: int = 0) -> list[dict]:
    if not indices:
        return []
    gt = gap_thresh or settings.GAP_THRESHOLD
    segs = []
    cur, cur_sc = [indices[0]], [scores[0]]
    for i in range(1, len(indices)):
        if indices[i] - indices[i-1] <= gt:
            cur.append(indices[i]); cur_sc.append(scores[i])
        else:
            segs.append({"frame_indices": cur, "scores": cur_sc})
            cur, cur_sc = [indices[i]], [scores[i]]
    if cur:
        segs.append({"frame_indices": cur, "scores": cur_sc})
    return segs

def parse_query(query: str) -> dict:
    m = re.search(r"between\s+([\d:]+(?:\s*[AP]M)?)\s+(?:and|to)\s+([\d:]+(?:\s*[AP]M)?)", query, re.IGNORECASE)
    return {"semantic_query": query, "time_range": f"{m.group(1)} to {m.group(2)}" if m else None}

def extract_clip(video_path: str, start_time: float, end_time: float, output: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    end = int(end_time * fps)
    cur = int(start_time * fps)
    while cur < end:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        cur += 1
    cap.release()
    out.release()

def index_video(video_path: str, video_id: str, progress: dict = None) -> VideoIndex:
    t0 = time.time()
    cap_info = cv2.VideoCapture(video_path)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_info.release()
    if progress is not None:
        progress["_t0"] = t0
        progress["stage"] = "motion_scan"
        progress["percent"] = 0
        progress["message"] = "Starting motion scan..."
    keep_frames, timestamps, motion_scores, total = motion_sample(str(video_path), progress=progress)

    min_keyframes = int(total / max(fps, 1) * settings.INDEX_MIN_FPS)
    if len(keep_frames) < min_keyframes:
        logger.info("Enforcing min FPS: adding %d uniform frames to meet INDEX_MIN_FPS=%d",
                    min_keyframes - len(keep_frames), settings.INDEX_MIN_FPS)
        existing = set(keep_frames)
        uniform = [int(f) for f in np.linspace(0, total - 1, min_keyframes, dtype=int) if f not in existing]
        uniform = uniform[:min_keyframes - len(keep_frames)]
        for f in uniform:
            keep_frames.append(f)
            timestamps.append(f / fps)
            motion_scores.append(0.0)
        combined = sorted(zip(keep_frames, timestamps, motion_scores), key=lambda x: x[0])
        keep_frames = [c[0] for c in combined]
        timestamps = [c[1] for c in combined]
        motion_scores = [c[2] for c in combined]

    t1 = time.time()
    if progress is not None:
        progress["stage"] = "extract"
        progress["percent"] = 20
        progress["message"] = f"Extracting {len(keep_frames)} keyframes..."
    logger.info("Frame extraction (%d frames)...", len(keep_frames))
    frames = extract_frames_parallel(str(video_path), keep_frames)
    t2 = time.time()
    if progress is not None:
        progress["stage"] = "embed"
        progress["percent"] = 25
        progress["message"] = f"Embedding {len(frames)} frames..."
    logger.info("CLIP embeddings (%d frames)...", len(frames))
    embs = compute_embeddings(frames, progress=progress)
    t3 = time.time()

    # ── Phase 4: build overlapping clip embeddings (parallel) ──
    clip_indices: list[list[int]] = []
    clip_embs_list: list[np.ndarray] = []
    w = settings.CLIP_WINDOW_SIZE
    stride = settings.CLIP_STRIDE
    r = range(0, max(1, len(keep_frames) - w + 1), stride)

    def _clip_embed(i: int) -> tuple[list[int], np.ndarray]:
        win_frame_idxs = keep_frames[i:i + w]
        win_embs = embs[i:i + w]
        win_motion = motion_scores[i:i + w]
        return win_frame_idxs, _encode_clip_light(win_embs, win_motion)

    rlist = list(r)
    if len(rlist) > 4:
        with ThreadPoolExecutor(max_workers=min(settings.NUM_WORKERS, len(rlist))) as pool:
            results = list(pool.map(_clip_embed, rlist))
    else:
        results = [_clip_embed(i) for i in rlist]
    for win_frame_idxs, clip_emb in results:
        clip_indices.append(win_frame_idxs)
        clip_embs_list.append(clip_emb)
    clip_embeddings_arr = np.array(clip_embs_list, dtype="float32") if clip_embs_list else np.empty((0, 515), dtype="float32")
    t_clip = time.time()
    if len(clip_embeddings_arr) > 0:
        logger.info("Clip embeddings: %d clips from %d frames in %.1fs",
                    len(clip_embeddings_arr), len(keep_frames), t_clip - t3)

    captions: dict[int, str] = {}
    caption_embs: np.ndarray | None = None
    t_cap = t3
    if settings.CAPTIONER_ENABLED:
        if progress is not None:
            progress["stage"] = "caption"
            progress["message"] = f"Generating captions for {len(frames)} keyframes..."
        logger.info("Captioning %d keyframes with %s...", len(frames), settings.CAPTIONER_MODEL)
        captioner = _get_captioner()
        if captioner is not None:
            caption_texts: list[str] = []
            cap_frames = list(zip(keep_frames, frames))
            nw = min(settings.NUM_WORKERS, len(cap_frames))
            if nw > 1:
                cap_results: list[tuple[int, str | None]] = [(f[0], None) for f in cap_frames]
                with ThreadPoolExecutor(max_workers=nw) as pool:
                    fut_map = {pool.submit(captioner.caption, frame): i
                               for i, (_, frame) in enumerate(cap_frames)}
                    done = 0
                    for f in as_completed(fut_map):
                        i = fut_map[f]
                        try:
                            cap_results[i] = (cap_frames[i][0], f.result())
                        except Exception:
                            logger.exception("Caption failed for frame %d", cap_frames[i][0])
                        done += 1
                        if progress is not None and done % max(1, len(cap_frames) // 5) == 0:
                            progress["message"] = f"Captioned: {done}/{len(cap_frames)} frames"
                for frame_idx, cap in cap_results:
                    if cap is not None:
                        captions[frame_idx] = cap
                        caption_texts.append(cap)
            else:
                for fi_idx, (frame_idx, frame) in enumerate(cap_frames):
                    try:
                        cap = captioner.caption(frame)
                        captions[frame_idx] = cap
                        caption_texts.append(cap)
                    except Exception:
                        logger.exception("Caption failed for frame %d", frame_idx)
                    if progress is not None and (fi_idx + 1) % max(1, len(frames) // 5) == 0:
                        progress["message"] = f"Captioned: {fi_idx+1}/{len(frames)} frames"
            if caption_texts:
                logger.info("Encoding %d captions...", len(caption_texts))
                caption_embs = embed_text(caption_texts)
    t_cap = time.time()

    object_metadata: list[dict] = []
    object_crops: list[np.ndarray] = []
    object_embs: np.ndarray = np.empty((0, settings.INDEX_OBJECT_EMBED_DIM), dtype="float32")
    t_od = t3
    t_oe = t3
    if progress is not None:
        progress["stage"] = "detect_objects"
        progress["message"] = f"Detecting objects in {len(frames)} keyframes..."
    logger.info("Phase 1+2: detecting + tracking objects in %d keyframes...", len(frames))
    t_od = time.time()
    tracker = SimpleTracker(match_thresh=settings.TRACK_MATCH_THRESHOLD,
                             track_buffer=settings.TRACK_BUFFER)
    # Phase 1: detect all frames in parallel
    all_dets: list[list[dict]] = [None] * len(frames)  # type: ignore
    nw = min(settings.NUM_WORKERS, len(frames))
    if nw > 1:
        with ThreadPoolExecutor(max_workers=nw) as pool:
            fut_map = {pool.submit(detect, frame, settings.INDEX_DETECTION_PROMPTS,
                                   threshold=settings.INDEX_OBJECT_CONFIDENCE): i
                       for i, (_, frame) in enumerate(zip(keep_frames, frames))}
            for f in as_completed(fut_map):
                i = fut_map[f]
                try:
                    all_dets[i] = f.result()
                except Exception:
                    all_dets[i] = []
    else:
        for i, (_, frame) in enumerate(zip(keep_frames, frames)):
            try:
                all_dets[i] = detect(frame, settings.INDEX_DETECTION_PROMPTS,
                                     threshold=settings.INDEX_OBJECT_CONFIDENCE)
            except Exception:
                all_dets[i] = []
    # Phase 2: track sequentially in frame order
    for fi_idx, (frame_idx, frame) in enumerate(zip(keep_frames, frames)):
        dets = all_dets[fi_idx] or []
        tracked = tracker.update(dets, frame_idx)
        for d in tracked:
            x1, y1, x2, y2 = d["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            object_metadata.append({
                "frame_idx": frame_idx,
                "bbox": d["bbox"],
                "label": d["label"],
                "score": d["score"],
                "track_id": d["track_id"],
            })
            object_crops.append(crop)
        if progress is not None and (fi_idx + 1) % max(1, len(frames) // 5) == 0:
            progress["message"] = f"Detected+tracked: {fi_idx+1}/{len(frames)} frames, {len(object_metadata)} objects"
    track_metadata = tracker.summary()
    t_od_end = time.time()
    if object_crops:
        logger.info("CLIP encoding %d object crops...", len(object_crops))
        object_embs = compute_embeddings(object_crops)
    t_oe = time.time()
    logger.info("Object detection: %d objects in %.1fs, embedding: %.1fs",
                len(object_metadata), t_od_end - t_od, t_oe - t_od_end)

    mdb_dir = STORAGE / "frames" / video_id
    mdb_dir.mkdir(parents=True, exist_ok=True)
    mdb_path = mdb_dir / "metadata.db"
    try:
        mdb = MetadataDB(mdb_path)
        mdb.populate(object_metadata, track_metadata, keep_frames, timestamps)
        mdb.close()
    except Exception:
        logger.exception("Failed to create metadata DB for %s", video_id)

    idx = VideoIndex(video_id=video_id, frame_indices=keep_frames, timestamps=timestamps,
                     motion_scores=motion_scores, total_frames=total,
                     metadata={"fps": round(fps, 2), "width": width, "height": height, "duration": round(total / max(fps, 1), 2)},
                     benchmarks={"scan_time": round(t1 - t0, 2), "extract_time": round(t2 - t1, 2),
                                 "embed_time": round(t3 - t2, 2), "detection_time": round(t_oe - t3, 2),
                                 "caption_time": round(t_cap - t3, 2),
                                 "total_time": round(t_oe - t0, 2)},
                     object_metadata=object_metadata, track_metadata=track_metadata,
                     captions=captions, clip_indices=clip_indices)

    t4 = time.time()
    out_dir = STORAGE / "frames" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build all FAISS indexes in parallel (they are independent)
    faiss_tasks: list[tuple[str, callable]] = [
        ("main", lambda: (build_faiss_index(embs), idx.save_faiss_index, idx.save_embeddings)),
    ]
    if len(object_embs) > 0:
        faiss_tasks.append(("object", lambda: (build_faiss_index(object_embs), idx.save_object_faiss_index, idx.save_object_embeddings)))
    if caption_embs is not None and len(caption_embs) > 0:
        faiss_tasks.append(("caption", lambda: (build_faiss_index(caption_embs), idx.save_caption_faiss_index, idx.save_caption_embeddings)))

    def _build_one(name: str, fn: callable):
        index, save_index, save_embs = fn()
        save_index(index)
        save_embs(None if name == "main" else (object_embs if name == "object" else caption_embs))
        return name

    # Actually, let me do a simpler pattern: build indexes in parallel, save sequentially after
    def _build_main():
        fi = build_faiss_index(embs)
        idx.save_faiss_index(fi)
        idx.save_embeddings(embs)
        return "main"
    def _build_object():
        fi = build_faiss_index(object_embs)
        idx.save_object_faiss_index(fi)
        idx.save_object_embeddings(object_embs)
        return "object"
    def _build_caption():
        fi = build_faiss_index(caption_embs)
        idx.save_caption_faiss_index(fi)
        idx.save_caption_embeddings(caption_embs)
        return "caption"

    fns = [_build_main]
    if len(object_embs) > 0:
        fns.append(_build_object)
    if caption_embs is not None and len(caption_embs) > 0:
        fns.append(_build_caption)

    if len(fns) > 1:
        with ThreadPoolExecutor(max_workers=len(fns)) as pool:
            list(pool.map(lambda f: f(), fns))
    else:
        fns[0]()

    tf = time.time()
    logger.info("FAISS index build: %.1fs", tf - t4)
    if progress is not None:
        progress["percent"] = 88
        progress["message"] = "Built FAISS index..."

    if len(clip_embeddings_arr) > 0:
        idx.save_clip_embeddings(clip_embeddings_arr)
        logger.info("Clip embeddings saved: %d clips in %.1fs",
                    len(clip_embeddings_arr), time.time() - t_clip)

    if progress is not None:
        progress["stage"] = "save"
        progress["percent"] = 90
        progress["message"] = f"Saving {len(keep_frames)} thumbnails..."
    def _save(idx_frame):
        i, f_idx = idx_frame
        cv2.imwrite(str(out_dir / f"frame_{f_idx}.jpg"), frames[i], [int(cv2.IMWRITE_JPEG_QUALITY), settings.THUMBNAIL_QUALITY])
    with ThreadPoolExecutor(max_workers=settings.NUM_WORKERS) as pool:
        pool.map(_save, enumerate(keep_frames))

    idx_dict = asdict(idx)
    idx_dict.pop("embeddings", None)
    idx_dict["embeddings_path"] = str(idx.embeddings_path())
    with open(out_dir / "index.json", "w") as f:
        json.dump(idx_dict, f)

    if progress is not None:
        progress["stage"] = "done"
        progress["percent"] = 100
        progress["message"] = "Complete"
        progress["keyframes"] = len(keep_frames)
        progress["total_frames"] = total
    bm.record_index(t1 - t0, t2 - t1, t3 - t2, t_oe - t0, total, len(keep_frames))
    logger.info("Times: scan=%.1fs, extract=%.1fs, embed=%.1fs, faiss=%.1fs, detection=%.1fs, total=%.1fs",
                t1 - t0, t2 - t1, t3 - t2, tf - t4, t_oe - t3, t_oe - t0)
    n_tracks = len(track_metadata)
    logger.info("Indexed %s: %d keyframes, %d objects, %d tracks, from %d total frames",
                video_id, len(keep_frames), len(object_metadata), n_tracks, total)
    return idx
