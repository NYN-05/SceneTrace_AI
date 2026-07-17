# SceneTrace AI — Project Documentation

> Verified Natural Language Video Grounding and Intelligent Clip Extraction
> Team Return0

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Usage Guide](#4-usage-guide)
5. [Pipeline Details](#5-pipeline-details)
6. [API Reference](#6-api-reference)
7. [Performance Optimizations](#7-performance-optimizations)
8. [Project Structure](#8-project-structure)
9. [Metrics](#9-metrics)
10. [Future Scope](#10-future-scope)

---

## 1. Overview

### Problem Statement

Develop a natural language interface capable of extracting relevant video clips from recorded footage based on textual descriptions. Instead of manually reviewing hours of footage, the system automatically retrieves the most relevant clip along with evidence supporting the match.

### One-Line Pitch

SceneTrace AI transforms recorded videos into searchable visual evidence by allowing users to describe events in natural language and retrieving the most relevant clips with refined timestamps, confidence scores, and structured evidence.

### Core Features

| Feature | Description |
|---------|-------------|
| Natural Language Search | Type plain English queries instead of browsing videos manually |
| Semantic Video Retrieval | CLIP embeddings enable concept-level matching, not keyword search |
| Zero-Shot Object Localization | YOLO-World-L (auto fallback M→S) or Grounding DINO detects arbitrary objects from query text |
| Weighted Explainable Scoring | 55% semantic + 45% object match with full score breakdown and explanation |
| Live Progress Tracking | Real-time progress bar with stage name, percentage, and ETA during indexing |
| Confidence Gating | Results classified as HIGH (>0.25), MEDIUM (>0.15), or LOW confidence |
| Performance Dashboard | Live metrics: indexing speed (fps), avg query latency, frame reduction %, GPU status |
| Event Timeline | Per-video frame timeline with motion scores and thumbnail previews |
| Evidence Reports | Per-video reports with frame reduction %, motion activity, and keyframe count |
| Index Persistence | Indexes survive server restarts via JSON + FAISS binary serialization + disk reload |

---

## 2. Architecture

### System Flow

```
                    INDEXING PIPELINE (Async)

1. User uploads video → FastAPI validates format, saves to storage/originals/
2. Motion-guided sparse sampling (160×90, stride=3, adaptive percentile)
3. Parallel frame extraction (4 worker threads, each with own VideoCapture)
4. Batched CLIP embeddings (batch size 32, GPU, 512-dim vectors)
5. FAISS index built (IVFFlat for large sets, FlatIP for small)
6. Thumbnails + index.json + metadata/benchmarks saved to disk
7. UI polls /index-progress every 800ms → real-time progress bar

                    ENHANCED SEARCH PIPELINE

8.  User enters natural language query
9.  CLIP text embedding + FAISS search → candidate frame indices
10. Frame-index gap clustering → initial segments
11. YOLO-World-L object detection on middle frame of each segment
    (auto falls back to YOLO-World-M → YOLO-World-S on failure)
12. Weighted scoring: 55% semantic (CLIP) + 45% object match (YOLO-World)
13. Score breakdown + explanation generated for each segment
14. Rich search cards displayed: thumbnails, bboxes, labels, score bars, explanation
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Motion Analysis | OpenCV Farneback Optical Flow (160×90) | 200+ FPS motion saliency at 4x reduced resolution |
| Frame Sampling | Adaptive percentile threshold (stride=3) | Auto-tunes to keep ~5% of frames per video |
| Parallel I/O | ThreadPoolExecutor (4 workers) | Concurrent frame extraction |
| Vision-Language | CLIP ViT-B/32 (HuggingFace Transformers) | Batched GPU inference, 512-dim embeddings |
| Open-Vocab Detection | YOLO-World-L with auto fallback (L→M→S) | Zero-shot object localization from query text |
| Weighted Scoring | Search Engine module | 55% CLIP semantic + 45% object match |
| Vector Search | FAISS IndexIVFFlat | O(log n) approximate nearest neighbor search |
| Segment Clustering | Frame-index gap threshold | Groups consecutive high-scoring frames |
| Performance Tracking | BenchmarkStats singleton (in config.py, thread-safe) | Real-time indexing speed, query latency, frame reduction |
| Backend | FastAPI + ThreadPoolExecutor | Async endpoints, CPU tasks offloaded, 14 endpoints |
| Frontend | React + Vite + Tailwind CSS | Google-like search, rich cards, dashboard, timeline |
| Storage | Local filesystem (JSON + FAISS binary) | Index persistence across restarts |

---

## 3. Installation & Setup

### Prerequisites

- **Python 3.10+** with: torch, opencv-python, transformers, faiss-cpu, fastapi, uvicorn, numpy, Pillow, python-multipart, python-dotenv, httpx, pytest, ultralytics
- **Node.js 18+** with npm
- **CUDA-capable GPU** recommended (falls back to CPU)

### Quick Start

**Terminal 1 — Backend:**
```powershell
cd backend
pip install -r requirements.txt
python main.py
# Server at http://localhost:8000
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

**Stop servers:**
```powershell
.\stop_servers.ps1
```

**Test via CLI (servers must be running):**
```powershell
.\test_workflow.ps1 -VideoPath test.mp4
```

**Unit tests (no server needed):**
```powershell
python -m pytest backend\tests\ -v
# 37 tests covering pipeline, search, benchmark, config, and API
```

> Optional: copy `backend/.env.example` to `backend/.env` to customize detector backend, CORS origins, upload size limit, etc.

---

## 4. Usage Guide

### Using the Web UI

1. Open `http://localhost:5173` in a browser
2. **Search tab** (default): Large centered search bar with example query chips below
   - Type a query → autocomplete suggestions appear from `/api/search/suggest`
   - Click a suggestion chip or press Enter
   - Rich result cards show: thumbnail (with bounding boxes if objects detected), confidence badge, timestamp, match %, **score breakdown bar** (semantic + object match), **detected object chips**, **explanation panel**
   - Buttons: Copy Timestamp, Download Clip
3. **Upload tab:** Click file picker → select `.mp4` → click "Upload & Index"
   - **Live progress bar** with stage name, percentage, ETA
   - Completes with: "Ready — N keyframes indexed"
4. **Dashboard tab:** Performance metrics grid
   - Indexing Speed (fps), Videos Indexed, Total Frames, Keyframes Kept, Frame Reduction %, Avg Query Latency, GPU status, Uptime
   - Click "Refresh" for latest data
5. **Timeline tab:** Select a video to view its event timeline
   - Visual timeline with motion score markers
   - Keyframe thumbnail grid

### Detector Backend Configuration

The system defaults to **YOLO-World-L** for zero-shot object detection. To switch to **Grounding DINO**, set in `.env`:
```
DETECTOR_BACKEND=grounding_dino
```

YOLO-World automatically falls back through Medium → Small if the Large variant fails due to OOM, timeout, or any runtime error. All fallbacks are logged and transparent to the user.

### Time Estimates

| Video | Frames | Motion Scan | CLIP Embed | Total Index |
|-------|--------|-------------|------------|-------------|
| 5 sec (test) | 90 | < 1s | ~13s | ~13s |
| 8 min (real) | 14,400 | ~30-60s | ~2-5 min | ~3-6 min |
| 30 min (typical) | 54,000 | ~2-4 min | ~5-15 min | ~10-20 min |

First index is slowest — CLIP model (~600MB) loads into GPU memory on first request.

---

## 5. Pipeline Details

### 5.1 Motion-Guided Sparse Sampling

Scans video frames using frame differencing (default, 3-5x faster) or Farneback optical flow to detect motion, keeping only frames with significant activity.

```
For each stride-sampled frame (every 3rd frame):
  1. Downscale to 160x90 grayscale (4x fewer pixels)
  2. Compute motion score via frame differencing or Farneback
  3. Sum flow magnitude → motion score
  4. After full scan: compute percentile threshold (keeps top ~5%)
  5. Extract surviving frames in parallel (4 threads)
```

**Optimizations:**
- **Frame differencing**: 3-5x faster than Farneback, equivalent keyframe quality
- **Stride=3**: 66% fewer computations
- **160x90 resolution**: 4x fewer pixels vs 320x180
- **Adaptive percentile**: No manual threshold tuning needed
- **Parallel extraction**: 3-4x faster than sequential

### 5.2 CLIP Embedding Generation

Each keyframe is encoded into a 512-dimensional vector using OpenAI's CLIP ViT-B/32 model.

- Batched inference: 32 frames per GPU batch
- CUDA acceleration for both image and text encoding
- Progress logged on every 160 frames (5 batches)

### 5.3 Vector Search (FAISS)

| Index Type | When Used | Search Complexity |
|-----------|-----------|-------------------|
| IndexFlatIP | < 50 vectors | O(n) brute force |
| IndexIVFFlat | ≥ 50 vectors | O(log n) approximate |

IVFFlat parameters: `nlist = sqrt(n)`, `nprobe = nlist / 4`

**Pre-built FAISS index:** The index is built once during `index_video` and serialized to a `.faiss` file alongside the JSON metadata. On server restart, indexes are loaded on demand. This eliminates the O(n·d) rebuild cost on every search query — search latency drops from ~50ms to ~1ms per query.

### 5.4 Segment Clustering

Consecutive frame indices with a gap ≤ 3 are grouped into a single segment. Each segment gets an average confidence score. Segments are sorted by score descending.

### 5.5 Async Indexing with Live Progress

Indexing runs in a **background thread** via `ThreadPoolExecutor` — the `/index` endpoint returns immediately, and the frontend polls `/index-progress` every 800ms for real-time updates.

**Progress stages mapped to percentage:**
| Stage | Percent | What happens |
|-------|---------|-------------|
| Motion scan | 0–18% | Frame differencing or Farneback flow on every 3rd frame |
| Frame extraction | 18–25% | Parallel extraction of surviving keyframes (4 threads) |
| CLIP embedding | 25–85% | Batched GPU inference, updates per batch |
| FAISS index build | 85–88% | Pre-built vector index serialized to disk |
| Saving | 88–100% | Thumbnails written + index.json serialized |

Each update includes `stage`, `percent`, `message` (human-readable stage description), and `eta_seconds` (estimated time remaining). The frontend displays an animated gradient progress bar with this data.

### 5.6 Persistence

After indexing, the full `VideoIndex` dataclass (frame_indices, timestamps, embeddings, motion_scores, total_frames) is serialized to `storage/frames/{video_id}/index.json`. The FAISS index is saved to `storage/frames/{video_id}/index.faiss`. On server startup, the lifespan handler reloads all saved indexes from disk.

---

## 6. API Reference

All endpoints are prefixed with `/api`. The frontend Vite dev server proxies `/api` to `localhost:8000`.

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/api/health` | Server status | — | `{status, indexed_videos}` |
| POST | `/api/videos/upload` | Upload video | multipart file | `{video_id, filename, size}` |
| POST | `/api/videos/{id}/index` | Start async indexing | — | `{video_id, status: "indexing_started"}` |
| GET | `/api/videos/{id}/index-progress` | Poll indexing progress | — | `{stage, percent, message, eta_seconds}` |
| GET | `/api/videos/{id}/status` | Index readiness | — | `{video_id, keyframes, timestamps, status}` |
| GET | `/api/videos/{id}/timeline` | Event timeline | — | `{video_id, metadata, events[]}` |
| GET | `/api/videos/{id}/objects` | Detected objects | — | `{video_id, detected_frames[]}` |
| POST | `/api/search` | Semantic search (v1, compat) | `{query, top_k}` | `{video_id, segments[], status}` |
| POST | `/api/v2/search` | Enhanced search w/ detection | `{query, top_k, enable_detection}` | `{segments[], score_breakdown, query_time}` |
| POST | `/api/search/suggest` | Query suggestions | `{query}` | `{suggestions[]}` |
| GET | `/api/dashboard/metrics` | Performance dashboard | — | `{indexing_speed_fps, videos_indexed, avg_query_latency, ...}` |
| GET | `/api/clips/{id}` | Download clip | query: `start_frame, end_frame` | MP4 file |
| GET | `/api/reports/{id}` | Generate report | — | `{video_id, keyframes_count, frame_reduction_pct}` |
| GET | `/api/metrics` | Basic metrics | — | `{videos_indexed, total_keyframes}` |

---

## 7. Performance Optimizations

### Implemented Optimizations

| Area | Before | After | Speedup |
|------|--------|-------|---------|
| Motion scan | Farneback only | Frame differencing (default) + Farneback opt-in | 3-5x faster |
| Motion scan resolution | 320x180 (57,600 px) | 160x90 (14,400 px) | 4x fewer pixels |
| Flow computation | Every frame | Every 3rd frame (stride=3) | 66% fewer computations |
| Frame selection | Fixed threshold (0.3) | Adaptive percentile (top 5%) | No manual tuning |
| Frame extraction | Single-threaded sequential | 4 threads parallel | 3-4x faster |
| CLIP inference | Unbatched | Batch size 32 | Up to 32x throughput |
| FAISS index build | Per-query rebuild | Pre-built once during indexing | ~50ms → ~1ms per query |
| Thumbnail saving | Sequential loop | ThreadPoolExecutor.map | 4x faster |
| Model inference | `@torch.no_grad()` | `@torch.inference_mode()` | 2-5% faster |

### Enhanced Search Optimizations

| Area | Technique | Benefit |
|------|-----------|---------|
| Object detection | YOLO-World-L on middle frame of each segment only | Avoids running on all frames — constant cost per query |
| Detection caching | Results cached per `(video_id, frame_index)` across queries | Repeated queries hit cache instantly |
| Lazy model loading | YOLO-World loads on first search, not at server start | Zero memory cost until detection is actually used |
| Auto fallback | YOLO-World-L → M → S on failure | No user intervention needed on OOM/error |
| Weighted scoring | 55% CLIP + 45% object match | Better ranking without increasing latency |
| Benchmark tracking | Thread-safe singleton with lock-free reads | Zero overhead for metrics collection |

### Bottleneck Analysis

| Stage | 5s video (90 frames) | 8min video (14,400 frames) | Scaling |
|-------|---------------------|---------------------------|---------|
| Motion scan | < 1s | ~30-60s | Linear with frame count |
| Frame extraction | < 1s | ~10-20s | Linear with keyframes |
| CLIP embedding | ~13s | ~2-5 min | Linear with keyframes |
| **Total** | **~13s** | **~3-6 min** | — |

The CLIP embedding stage is the dominant bottleneck for long videos. It scales linearly with the number of keyframes (~100ms per 32-frame batch on GPU).

### Security Hardening

| Measure | Implementation | Benefit |
|---------|---------------|---------|
| CORS restriction | `CORS_ORIGINS` env var (default: `http://localhost:5173`) | Prevents cross-origin data exfiltration |
| File size limit | `MAX_UPLOAD_MB` env var (default: 500 MB) | Prevents DoS via oversized uploads |
| Input validation | Pydantic `Field(ge=1, le=100)` on `top_k`, `max_length=500` on query | Blocks malformed requests at the boundary |
| Extension whitelist | `{".mp4", ".avi", ".mov", ".mkv", ".webm"}` | Prevents arbitrary file upload |
| Empty file rejection | `len(content) == 0` check | Avoids processing zero-byte files |
| Silent error handling | `except` without stack trace display | Prevents information leakage |
| Environment isolation | `config.py` + `.env.example` | No hardcoded secrets in source |

---

### Dashboard Metrics Collected

| Metric | Source | Why it matters |
|--------|--------|---------------|
| Indexing speed (fps) | `benchmark.record_index()` | Measures pipeline throughput |
| Avg query latency | `benchmark.record_query()` | Measures search responsiveness |
| Frame reduction % | Total frames vs keyframes | Quantifies motion sampling efficiency |
| GPU availability | `torch.cuda.is_available()` | Confirms hardware acceleration |
| Total index time | Cumulative per-video timing | Tracks overall processing cost |
| Query count | Running total | Measures system usage |
| Uptime | Server start timestamp | Shows reliability |

---

## 8. Project Structure

```
AUTOMATE/
├── .gitignore
├── backend/
│   ├── main.py              # FastAPI server (16 endpoints + security + async indexing)
│   ├── pipeline.py          # CV pipeline (motion, CLIP, FAISS, extraction, progress, benchmarks)
│   ├── search_engine.py     # Enhanced search: detection integration + weighted scoring + suggestions
│   ├── detector.py          # YOLO-World (L→M→S fallback) + Grounding DINO (lazy-loaded, GPU)
│   ├── config.py            # Environment config + BenchmarkStats singleton
│   ├── preload_models.py    # Pre-download all ML models (CLIP, YOLO-World, DINO)
│   ├── requirements.txt     # Pinned Python dependencies
│   ├── pytest.ini           # Test configuration
│   ├── .env.example         # Environment variable template
│   ├── checkpoints/         # Downloaded model weight files
│   ├── tests/               # 37 pytest tests (unit + API)
│   │   ├── test_pipeline.py
│   │   ├── test_search.py
│   │   ├── test_benchmark.py
│   │   ├── test_config.py
│   │   └── test_api.py
│   └── storage/             # Local data storage
│       ├── originals/       # Uploaded video files
│       ├── frames/          # Keyframe thumbnails + index.json + index.faiss
│       ├── clips/           # Extracted MP4 clips
│       └── reports/         # Generated JSON reports
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main SPA (slim — imports components)
│   │   ├── main.jsx         # Entry point
│   │   ├── index.css        # Tailwind imports
│   │   ├── components/      # 8 modular components
│   │   │   ├── SearchTab.jsx
│   │   │   ├── UploadTab.jsx
│   │   │   ├── DashboardTab.jsx
│   │   │   ├── TimelineTab.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── ProgressBar.jsx
│   │   │   ├── ScoreBar.jsx
│   │   │   └── MetricCard.jsx
│   │   └── hooks/
│   │       └── useApi.js    # Custom hooks for search + upload
│   ├── index.html           # HTML shell
│   ├── vite.config.js       # Vite config with /api proxy
│   ├── package.json         # Dependencies (React, Vite, Tailwind)
│   ├── postcss.config.js
│   └── tailwind.config.js
├── Docs/
│   ├── SceneTrace_AI_Final_Idea.md  # This document
│   ├── SceneTrace_AI_Hackathon_Doc.docx
│   ├── TECHNICAL_BRIEF.md   # Full project justification
│   └── PS.png
├── stop_servers.ps1         # Kill processes on ports 8000 and 5173
├── test_workflow.ps1        # 12-step automated test suite (all endpoints)
└── README.md                # Quick-start guide
```

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `backend/detector.py` | ~185 | YOLO-World-L (→M→S fallback) + Grounding DINO: lazy-loaded, `detect()` + `render()` for bboxes |
| `backend/search_engine.py` | ~134 | Enhanced search: CLIP + detection integration, weighted scoring, suggestions |
| `backend/config.py` | ~120 | Environment-based config + BenchmarkStats singleton (merged from benchmark.py) |
| `backend/preload_models.py` | ~65 | Pre-download all models (CLIP, YOLO-World, Grounding DINO) |
| `backend/pipeline.py` | ~346 | Core CV pipeline: motion, CLIP, FAISS, extraction, progress, metadata, benchmarks |
| `backend/main.py` | ~369 | FastAPI app: 16 endpoints, security hardening, async index, v2 search, dashboard |
| `frontend/src/App.jsx` | ~90 | App shell — imports 8 components + 2 hooks |
| `backend/tests/` | 5 files | 37 pytest tests (unit + API) covering all modules |
| `test_workflow.ps1` | ~120 | 12-step automated test suite covering all endpoints |

---

## 9. Metrics

| Metric | Measured Value | Method |
|--------|---------------|--------|
| Frame reduction (5s video) | 53-63% | Adaptive percentile motion sampling |
| Frame reduction (8min video) | ~97% | Same — longer video has more static frames |
| Motion scan speed | ~30s for 14K frames | 160x90 gray, stride=3, 200+ FPS effective |
| CLIP embedding throughput | ~300 frames/min | Batch size 32 on GPU |
| Query latency (v1) | < 5s | CLIP text embed + FAISS IVFFlat search |
| Enhanced search latency (w/ detection) | < 10s | v2 search with YOLO-World on top-K segments |
| Indexing speed | ~28 FPS | Real-time benchmark from dashboard metrics |
| Weighted scoring accuracy | 55% semantic + 45% object | Score breakdown per result |
| Object detection | Arbitrary text queries | YOLO-World-L zero-shot (no training) |
| Dashboard metrics | Live | Indexing speed, query latency, GPU, uptime |
| Format support | mp4, avi, mov, mkv, webm | OpenCV VideoCapture codec support |
| No-match rejection | Confidence-gated | Thresholds: high > 0.25, medium > 0.15 |
| Index persistence | Survives restarts | JSON + FAISS binary → lifespan handler reload |
| Max video tested | 8 min (177 MB, 4K) | 14,400 frames, ~400 keyframes retained |

---

## 10. Future Scope

| Feature | Status | Effort | Impact |
|---------|--------|--------|--------|
| ~~YOLO-World detection with fallback~~ | ✅ DONE | High | Unlimited object classes via YOLO-World-L (→M→S fallback) |
| ~~Open-vocabulary detection~~ | ✅ DONE | High | Unlimited object classes via Grounding DINO |
| Weighted explainable scoring | ✅ DONE | Medium | Score breakdown: 55% semantic + 45% object |
| Performance dashboard | ✅ DONE | Low | Real-time indexing speed, latency, GPU, reduction |
| Event timeline | ✅ DONE | Medium | Per-video frame timeline with motion scores |
| Search suggestions | ✅ DONE | Low | Autocomplete based on query patterns |
| Preload scripts for all models | ✅ DONE | Low | `python preload_models.py` downloads everything |
| Multi-object tracking (ByteTrack) | ❌ FUTURE | High | Eliminates single-frame false positives |
| Action recognition (X-CLIP) | ❌ FUTURE | Medium | Event-level understanding |
| Speech-to-text search (Whisper) | ❌ FUTURE | Low | Search by spoken content |
| Privacy blur (face detection) | ❌ FUTURE | Low | Responsible AI compliance |
| Cross-camera search | ❌ FUTURE | Medium | Multi-camera investigation |
| Real-time CCTV integration | ❌ FUTURE | High | Live video search |
| GPU FAISS (index on GPU) | ❌ FUTURE | Low | Faster search at 100K+ vectors |
| ONNX runtime for CLIP | ❌ FUTURE | Medium | Faster inference, no PyTorch dep |
| HNSW index instead of IVF | ❌ FUTURE | Low | Better accuracy/speed tradeoff |

---

*Documentation updated to reflect the implemented MVP. Originally conceived as a hackathon project for Team Return0.*
