import asyncio
import json
import uuid
import time
import cv2
import logging
import threading
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pipeline import (STORAGE, VideoIndex, search_embeddings,
                      frames_to_segments, parse_query, index_video, extract_clip, device)
from search_engine import search as enhanced_search, suggest as query_suggest
from config import benchmark
from config import settings, logger

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024

executor = ThreadPoolExecutor(settings.NUM_WORKERS)
_indexes: dict[str, VideoIndex] = {}
_index_progress: dict[str, dict] = {}
_indexes_lock = threading.Lock()
_index_progress_lock = threading.Lock()

api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)

limiter = Limiter(key_func=get_remote_address)

def authenticate(request: Request, api_key: str = Depends(api_key_header)):
    if not settings.API_KEY:
        return True
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True

def _load_persisted_indexes():
    frames_dir = STORAGE / "frames"
    if not frames_dir.exists():
        return
    for idx_dir in frames_dir.iterdir():
        idx_file = idx_dir / "index.json"
        if idx_file.exists():
            try:
                data = json.loads(idx_file.read_text())
                vi = VideoIndex(
                    video_id=data.get("video_id", idx_dir.name),
                    frame_indices=data.get("frame_indices", []),
                    timestamps=data.get("timestamps", []),
                    motion_scores=data.get("motion_scores", []),
                    total_frames=data.get("total_frames", 0),
                    metadata=data.get("metadata", {}),
                    benchmarks=data.get("benchmarks", {}),
                    object_metadata=data.get("object_metadata", []),
                    track_metadata=data.get("track_metadata", {}),
                )
                with _indexes_lock:
                    _indexes[vi.video_id] = vi
            except Exception:
                logger.exception("Failed loading persisted index from %s", idx_file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_persisted_indexes()
    if _indexes:
        logger.info("Reloaded %d indexes from disk (FAISS indexes loaded on demand)", len(_indexes))
    yield

app = FastAPI(title="SceneTrace AI", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=True,
)

frames_dir = STORAGE / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/frames", StaticFiles(directory=str(frames_dir)), name="frames")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=100)
    video_id: str | None = None

class V2SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=100)
    enable_detection: bool = True
    video_id: str | None = None

@app.get("/api/health")
def health():
    with _indexes_lock:
        count = len(_indexes)
        ids = list(_indexes.keys())
    return {"status": "ok", "indexed_videos": count, "video_ids": ids}

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...), _=Depends(authenticate)):
    ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")
    video_id = str(uuid.uuid4())[:8]
    dest = STORAGE / "originals" / f"{video_id}{ext}"
    size = 0
    with open(dest, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                buffer.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large. Max: {settings.MAX_UPLOAD_MB}MB")
            buffer.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file")
    try:
        import magic
        mime = magic.from_file(str(dest), mime=True)
        if not mime.startswith("video/"):
            dest.unlink(missing_ok=True)
            raise HTTPException(400, f"Invalid content type '{mime}'. Only video files are allowed.")
    except ImportError:
        pass
    filename_safe = file.filename.replace("\n", "").replace("\r", "").replace("\\", "").replace("/", "")
    logger.info("Uploaded %s (%d bytes) as %s", filename_safe, size, f"{video_id}{ext}")
    return {"video_id": video_id, "filename": file.filename, "size": size}

@app.post("/api/videos/{video_id}/index")
async def start_index(video_id: str, _=Depends(authenticate)):
    orig_dir = STORAGE / "originals"
    candidates = list(orig_dir.glob(f"{video_id}.*"))
    if not candidates:
        raise HTTPException(404, f"Video '{video_id}' not found in storage")
    video_path = candidates[0]
    logger.info("Starting index for %s (%s)...", video_id, video_path.name)
    with _index_progress_lock:
        _index_progress[video_id] = {"stage": "starting", "percent": 0, "message": "Queued...", "_t0": time.time()}

    def _do_index():
        import traceback
        try:
            with _index_progress_lock:
                prog = _index_progress.get(video_id, {"_t0": time.time()})
            idx = index_video(str(video_path), video_id, prog)
            with _indexes_lock:
                _indexes[video_id] = idx
            with _index_progress_lock:
                _index_progress.pop(video_id, None)
        except Exception as e:
            logger.exception("Indexing failed for %s", video_id)
            with _index_progress_lock:
                if video_id in _index_progress:
                    _index_progress[video_id].update({"stage": "error", "message": f"Failed: {str(e)}"})

    executor.submit(_do_index)
    return {"video_id": video_id, "status": "indexing_started"}

@app.get("/api/videos/{video_id}/index-progress")
def get_index_progress(video_id: str):
    with _index_progress_lock:
        p = _index_progress.get(video_id)
    if p is None:
        with _indexes_lock:
            exists = video_id in _indexes
            if exists:
                idx = _indexes[video_id]
        if exists:
            return {"stage": "done", "percent": 100, "message": "Complete",
                    "keyframes": len(idx.frame_indices), "total_frames": idx.total_frames}
        raise HTTPException(404, "No index in progress for this video")
    resp = dict(p)
    resp.pop("_t0", None)
    return resp

@app.get("/api/videos/{video_id}/status")
def video_status(video_id: str):
    with _indexes_lock:
        idx = _indexes.get(video_id)
    if not idx:
        raise HTTPException(404, "Video not indexed")
    return {"video_id": video_id, "keyframes": len(idx.frame_indices),
            "timestamps": idx.timestamps[:5], "total_frames": idx.total_frames, "status": "ready"}

async def _run_search_with_timeout(req: SearchRequest):
    with _indexes_lock:
        targets = {req.video_id: _indexes[req.video_id]} if req.video_id and req.video_id in _indexes else dict(_indexes)
    if not targets:
        return JSONResponse(status_code=400, content={"detail": "No videos indexed." if not req.video_id else f"Video {req.video_id} not found.", "segments": []})
    results = []
    for video_id, idx in targets.items():
        embs = idx.load_embeddings()
        if embs is None or len(embs) == 0:
            continue
        faiss_idx = idx.get_faiss_index()
        loop = asyncio.get_event_loop()
        indices, scores = await loop.run_in_executor(executor, search_embeddings, req.query, faiss_idx, embs, req.top_k * 2)
        segs = frames_to_segments(indices, scores)
        for seg in segs:
            orig = seg["frame_indices"]
            seg["video_id"] = video_id
            seg["frame_indices"] = [idx.frame_indices[i] for i in orig]
            seg["timestamps"] = [idx.timestamps[i] for i in orig]
            seg["avg_score"] = sum(seg["scores"]) / len(seg["scores"]) if seg["scores"] else 0
        segs.sort(key=lambda s: s["avg_score"], reverse=True)
        results.extend(segs[:req.top_k])
    results.sort(key=lambda s: s["avg_score"], reverse=True)
    results = results[:req.top_k]
    top_score = results[0]["avg_score"] if results else 0
    status_val = "high" if top_score > settings.SEARCH_HIGH_THRESHOLD else ("medium" if top_score > settings.SEARCH_MEDIUM_THRESHOLD else "low")
    with _indexes_lock:
        first_id = list(_indexes.keys())[0] if _indexes else ""
    return {"video_id": first_id, "segments": results, "query_info": parse_query(req.query), "status": status_val}

@app.post("/api/search")
async def search(req: SearchRequest, _=Depends(authenticate)):
    try:
        return await asyncio.wait_for(_run_search_with_timeout(req), timeout=settings.REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Search request timed out")

@app.get("/api/clips/{video_id}")
def get_clip(video_id: str, start_frame: int, end_frame: int, _=Depends(authenticate)):
    with _indexes_lock:
        idx = _indexes.get(video_id)
    if not idx:
        raise HTTPException(404, "Video not indexed")
    orig_dir = STORAGE / "originals"
    candidates = list(orig_dir.glob(f"{video_id}.*"))
    if not candidates:
        raise HTTPException(404, "Original video file not found")
    output = STORAGE / "clips" / f"{video_id}_{start_frame}_{end_frame}.mp4"
    cap = cv2.VideoCapture(str(candidates[0]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    start_idx = idx.frame_indices.index(start_frame) if start_frame in idx.frame_indices else -1
    end_idx = idx.frame_indices.index(end_frame) if end_frame in idx.frame_indices else -1
    start_time = idx.timestamps[start_idx] if start_idx >= 0 else start_frame / max(fps, 1)
    end_time = idx.timestamps[end_idx] if end_idx >= 0 else end_frame / max(fps, 1)
    extract_clip(str(candidates[0]), start_time, end_time, str(output))
    return FileResponse(str(output), media_type="video/mp4")

@app.get("/api/reports/{video_id}")
def generate_report(video_id: str, _=Depends(authenticate)):
    with _indexes_lock:
        idx = _indexes.get(video_id)
    if not idx:
        raise HTTPException(404, "Video not indexed")
    reduction = round((1 - len(idx.frame_indices) / max(idx.total_frames, 1)) * 100, 1)
    report = {"video_id": video_id, "keyframes_count": len(idx.frame_indices),
              "total_frames": idx.total_frames, "frame_reduction_pct": reduction,
              "motion_activity_avg": round(sum(idx.motion_scores) / max(len(idx.motion_scores), 1), 4)}
    (STORAGE / "reports" / f"{video_id}.json").write_text(json.dumps(report, indent=2))
    return report

@app.get("/api/metrics")
def metrics():
    with _indexes_lock:
        total_frames = sum(len(idx.frame_indices) for idx in _indexes.values())
        count = len(_indexes)
        ids = list(_indexes.keys())
    return {"videos_indexed": count, "total_keyframes": total_frames, "videos": ids}

async def _run_search_v2_with_timeout(req: V2SearchRequest):
    with _indexes_lock:
        targets = {req.video_id: _indexes[req.video_id]} if req.video_id and req.video_id in _indexes else dict(_indexes)
    if not targets:
        return JSONResponse(status_code=400, content={"detail": "No videos indexed." if not req.video_id else f"Video {req.video_id} not found.", "segments": []})
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, enhanced_search, req.query, targets, req.top_k, req.enable_detection
    )
    return result

@app.post("/api/v2/search")
async def search_v2(req: V2SearchRequest, _=Depends(authenticate)):
    try:
        return await asyncio.wait_for(_run_search_v2_with_timeout(req), timeout=settings.REQUEST_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(504, "Search request timed out")

@app.get("/api/videos/{video_id}/timeline")
def get_timeline(video_id: str):
    with _indexes_lock:
        idx = _indexes.get(video_id)
    if not idx:
        raise HTTPException(404, "Video not indexed")
    events = []
    for i, (fi, ts, ms) in enumerate(zip(idx.frame_indices, idx.timestamps, idx.motion_scores)):
        events.append({
            "frame_index": fi,
            "timestamp": round(ts, 2),
            "motion_score": round(ms, 4),
            "has_thumbnail": (STORAGE / "frames" / video_id / f"frame_{fi}.jpg").exists()
        })
    return {
        "video_id": video_id,
        "metadata": idx.metadata,
        "total_keyframes": len(idx.frame_indices),
        "events": events
    }

@app.get("/api/videos/{video_id}/objects")
def get_video_objects(video_id: str):
    with _indexes_lock:
        idx = _indexes.get(video_id)
    if not idx:
        raise HTTPException(404, "Video not indexed")
    vf = STORAGE / "frames" / video_id
    detected = []
    for fi in idx.frame_indices[:50]:
        annotated_path = vf / f"frame_{fi}_d.jpg"
        if annotated_path.exists():
            detected.append({
                "frame_index": fi,
                "timestamp": round(idx.timestamps[idx.frame_indices.index(fi)], 2) if fi in idx.frame_indices else 0,
                "annotated": f"/api/frames/{video_id}/frame_{fi}_d.jpg"
            })
    return {"video_id": video_id, "detected_frames": detected}

@app.get("/api/dashboard/metrics")
def dashboard_metrics():
    b = benchmark.stats()
    with _indexes_lock:
        b["indexed_videos"] = len(_indexes)
        b["videos"] = list(_indexes.keys())
        b["total_frames_motion"] = sum(len(idx.frame_indices) for idx in _indexes.values())
    b["gpu_available"] = (device == "cuda")
    return b

@app.post("/api/search/suggest")
def search_suggest(req: SearchRequest):
    suggestions = query_suggest(req.query)
    return {"query": req.query, "suggestions": suggestions}

@app.post("/api/storage/cleanup")
def cleanup_storage(_=Depends(authenticate)):
    cutoff = datetime.now() - timedelta(days=settings.STORAGE_CLEANUP_DAYS)
    cleaned = {"clips": 0, "reports": 0, "originals": 0}
    for dir_name, target_dir in [("clips", STORAGE / "clips"), ("reports", STORAGE / "reports"), ("originals", STORAGE / "originals")]:
        if target_dir.exists():
            for item in target_dir.iterdir():
                if item.is_file():
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    if mtime < cutoff:
                        item.unlink()
                        cleaned[dir_name] += 1
    logger.info("Storage cleanup removed: %s", cleaned)
    return {"cleaned": cleaned, "retention_days": settings.STORAGE_CLEANUP_DAYS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level=settings.LOG_LEVEL.lower())
