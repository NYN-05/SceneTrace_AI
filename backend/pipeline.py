import cv2
import numpy as np
import torch
from pathlib import Path
import faiss
import json
import re
import logging
import threading
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from benchmark import benchmark as bm
from config import settings, logger

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
                _clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(device).eval()
                _clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
                logger.info("CLIP model loaded")
    return _clip_model, _clip_processor

@torch.inference_mode()
def compute_embeddings(imgs: list[np.ndarray], batch_size: int = 0,
                       progress: dict = None) -> np.ndarray:
    model, processor = _get_clip()
    all_embs = []
    bs = batch_size or settings.BATCH_SIZE
    n_batches = max(1, (len(imgs) + bs - 1) // bs)
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i+bs]
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
        inputs = processor(images=rgb, return_tensors="pt", padding=True).to(device)
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
    return np.concatenate(all_embs, axis=0).astype("float32")

@torch.inference_mode()
def embed_text(texts: list[str]) -> np.ndarray:
    model, processor = _get_clip()
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

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
    idx = VideoIndex(video_id=video_id, frame_indices=keep_frames, timestamps=timestamps,
                     motion_scores=motion_scores, total_frames=total,
                     metadata={"fps": round(fps, 2), "width": width, "height": height, "duration": round(total / max(fps, 1), 2)},
                     benchmarks={"scan_time": round(t1 - t0, 2), "extract_time": round(t2 - t1, 2),
                                 "embed_time": round(t3 - t2, 2), "total_time": round(t3 - t0, 2)})

    t4 = time.time()
    faiss_index = build_faiss_index(embs)
    tf = time.time()
    logger.info("FAISS index build: %.1fs", tf - t4)
    if progress is not None:
        progress["percent"] = 88
        progress["message"] = "Built FAISS index..."

    out_dir = STORAGE / "frames" / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    idx.save_faiss_index(faiss_index)
    idx.save_embeddings(embs)

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
    bm.record_index(t1 - t0, t2 - t1, t3 - t2, t3 - t0, total, len(keep_frames))
    logger.info("Times: scan=%.1fs, extract=%.1fs, embed=%.1fs, faiss=%.1fs, total=%.1fs",
                t1 - t0, t2 - t1, t3 - t2, tf - t4, t3 - t0)
    logger.info("Indexed %s: %d keyframes from %d total frames", video_id, len(keep_frames), total)
    return idx
