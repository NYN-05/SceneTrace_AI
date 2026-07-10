import asyncio
import json
import uuid
import time
import cv2
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pipeline import (STORAGE, VideoIndex, search_embeddings,
                      frames_to_segments, parse_query, index_video, extract_clip)

executor = ThreadPoolExecutor(4)
_indexes: dict[str, VideoIndex] = {}
_index_progress: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    frames_dir = STORAGE / "frames"
    if frames_dir.exists():
        for idx_dir in frames_dir.iterdir():
            idx_file = idx_dir / "index.json"
            if idx_file.exists():
                try:
                    data = json.loads(idx_file.read_text())
                    _indexes[data["video_id"]] = VideoIndex(**data)
                except Exception as e:
                    print(f"Failed to reload {idx_file}: {e}")
        if _indexes:
            print(f"Reloaded {len(_indexes)} indexes from disk")
    yield

app = FastAPI(title="SceneTrace AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

frames_dir = STORAGE / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/frames", StaticFiles(directory=str(frames_dir)), name="frames")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    video_id: str
    segments: list
    query_info: dict
    status: str = "success"

@app.get("/api/health")
def health():
    return {"status": "ok", "indexed_videos": len(_indexes)}

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() if file.filename else ".mp4"
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {allowed}")
    video_id = str(uuid.uuid4())[:8]
    dest = STORAGE / "originals" / f"{video_id}{ext}"
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")
    dest.write_bytes(content)
    print(f"Uploaded {file.filename} ({len(content)} bytes) as {video_id}{ext}")
    return {"video_id": video_id, "filename": file.filename, "size": len(content)}

@app.post("/api/videos/{video_id}/index")
async def start_index(video_id: str):
    orig_dir = STORAGE / "originals"
    candidates = list(orig_dir.glob(f"{video_id}.*"))
    if not candidates:
        raise HTTPException(404, f"Video '{video_id}' not found in storage")
    video_path = candidates[0]
    print(f"Starting index for {video_id} ({video_path.name})...")

    _index_progress[video_id] = {"stage": "starting", "percent": 0, "message": "Queued...", "_t0": time.time()}

    def _do_index():
        import traceback
        try:
            idx = index_video(str(video_path), video_id, _index_progress[video_id])
            _indexes[video_id] = idx
        except Exception as e:
            traceback.print_exc()
            _index_progress[video_id].update({"stage": "error", "message": f"Failed: {str(e)}"})

    executor.submit(_do_index)
    return {"video_id": video_id, "status": "indexing_started"}

@app.get("/api/videos/{video_id}/index-progress")
def get_index_progress(video_id: str):
    p = _index_progress.get(video_id)
    if p is None:
        if video_id in _indexes:
            idx = _indexes[video_id]
            return {"stage": "done", "percent": 100, "message": "Complete",
                    "keyframes": len(idx.frame_indices), "total_frames": idx.total_frames}
        raise HTTPException(404, "No index in progress for this video")
    resp = dict(p)
    resp.pop("_t0", None)
    return resp

@app.get("/api/videos/{video_id}/status")
def video_status(video_id: str):
    idx = _indexes.get(video_id)
    if not idx:
        raise HTTPException(404, "Video not indexed")
    return {"video_id": video_id, "keyframes": len(idx.frame_indices),
            "timestamps": idx.timestamps[:5], "total_frames": idx.total_frames, "status": "ready"}

@app.post("/api/search")
async def search(req: SearchRequest):
    if not _indexes:
        return JSONResponse(status_code=400, content={"detail": "No videos indexed. Upload and index a video first.", "segments": []})
    results = []
    for video_id, idx in _indexes.items():
        embs = np.array(idx.embeddings, dtype="float32")
        if len(embs) == 0:
            continue
        loop = asyncio.get_event_loop()
        indices, scores = await loop.run_in_executor(executor, search_embeddings, req.query, embs, req.top_k * 2)
        segs = frames_to_segments(indices, scores)
        for seg in segs:
            seg["video_id"] = video_id
            seg["timestamps"] = [idx.timestamps[i] for i in seg["frame_indices"]]
            seg["avg_score"] = sum(seg["scores"]) / len(seg["scores"]) if seg["scores"] else 0
        segs.sort(key=lambda s: s["avg_score"], reverse=True)
        results.extend(segs[:req.top_k])
    results.sort(key=lambda s: s["avg_score"], reverse=True)
    results = results[:req.top_k]
    top_score = results[0]["avg_score"] if results else 0
    status = "high" if top_score > 0.25 else ("medium" if top_score > 0.15 else "low")
    return {"video_id": list(_indexes.keys())[0] if _indexes else "",
            "segments": results, "query_info": parse_query(req.query), "status": status}

@app.get("/api/clips/{video_id}")
def get_clip(video_id: str, start_frame: int, end_frame: int):
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
def generate_report(video_id: str):
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
    total_frames = sum(len(idx.frame_indices) for idx in _indexes.values())
    return {"videos_indexed": len(_indexes), "total_keyframes": total_frames,
            "videos": list(_indexes.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
