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
| Live Progress Tracking | Real-time progress bar with stage name, percentage, and ETA during indexing |
| Confidence Gating | Results classified as HIGH (>0.25), MEDIUM (>0.15), or LOW confidence |
| Evidence Reports | Per-video reports with frame reduction %, motion activity, and keyframe count |
| Index Persistence | Indexes survive server restarts via JSON serialization + disk reload |

---

## 2. Architecture

### System Flow

```
                    INDEXING PIPELINE

1. User uploads video → FastAPI validates format, saves to storage/originals/
2. Motion-guided sparse sampling (160×90, stride=3, adaptive percentile)
3. Parallel frame extraction (4 worker threads, each with own VideoCapture)
4. Batched CLIP embeddings (batch size 32, GPU, 512-dim vectors)
5. FAISS index built (IVFFlat for large sets, FlatIP for small)
6. Thumbnails + index.json saved to storage/frames/{video_id}/

                    QUERY PIPELINE

7. User enters natural language query
8. CLIP text embedding generated
9. FAISS index searched (nprobe approximation)
10. Frame-index gap clustering groups consecutive matches into segments
11. Confidence classified, top-K segments returned
12. Frontend displays cards with thumbnails, timestamps, match %
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Motion Analysis | OpenCV Farneback Optical Flow (160×90) | 200+ FPS motion saliency at 4× reduced resolution |
| Frame Sampling | Adaptive percentile threshold (stride=3) | Auto-tunes to keep ~5% of frames per video |
| Parallel I/O | ThreadPoolExecutor (4 workers) | Concurrent frame extraction |
| Vision-Language | CLIP ViT-B/32 (HuggingFace Transformers) | Batched GPU inference, 512-dim embeddings |
| Vector Search | FAISS IndexIVFFlat | O(log n) approximate nearest neighbor search |
| Segment Clustering | Frame-index gap threshold | Groups consecutive high-scoring frames |
| Backend | FastAPI + ThreadPoolExecutor | Async endpoints, CPU tasks offloaded |
| Frontend | React + Vite + Tailwind CSS | Upload dashboard + search interface |
| Storage | Local filesystem (JSON) | Index persistence across restarts |

---

## 3. Installation & Setup

### Prerequisites

- **Python 3.10+** with: torch, opencv-python, transformers, faiss-cpu, fastapi, uvicorn, numpy, Pillow, python-multipart, httpx
- **Node.js 18+** with npm
- **CUDA-capable GPU** recommended (falls back to CPU)

### Quick Start

**Terminal 1 — Backend:**
```powershell
cd backend
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

---

## 4. Usage Guide

### Using the Web UI

1. Open `http://localhost:5173` in a browser
2. **Upload tab:** Click file picker → select `.mp4` → click "Upload & Index"
   - Status shows: Uploading → Starting index...
   - **Live progress bar** appears with: animated gradient bar, stage name (Motion scan → Embedding → Saving), percentage, and ETA
   - Completes with: "Ready — N keyframes indexed"
3. **Search tab:** Type a query → press Enter or click "Search"
   - Example: `"a red circle moving"` or `"person near entrance"`
4. Results show:
   - Confidence badge: HIGH (green), MEDIUM (yellow), LOW (red)
   - Segment cards with frame thumbnails, timestamps, match percentage
   - Activity log shows all API interactions

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

Scans video frames using Farneback optical flow to detect motion, keeping only frames with significant activity.

```
For each stride-sampled frame (every 3rd frame):
  1. Downscale to 160×90 grayscale (4× fewer pixels)
  2. Compute Farneback optical flow against previous frame
  3. Sum flow magnitude → motion score
  4. After full scan: compute percentile threshold (keeps top ~5%)
  5. Extract surviving frames in parallel (4 threads)
```

**Optimizations:**
- **Stride=3**: 66% fewer flow computations
- **160×90 resolution**: 4× fewer pixels vs 320×180
- **Adaptive percentile**: No manual threshold tuning needed
- **Parallel extraction**: 3-4× faster than sequential

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

### 5.4 Segment Clustering

Consecutive frame indices with a gap ≤ 3 are grouped into a single segment. Each segment gets an average confidence score. Segments are sorted by score descending.

### 5.5 Async Indexing with Live Progress

Indexing runs in a **background thread** via `ThreadPoolExecutor` — the `/index` endpoint returns immediately, and the frontend polls `/index-progress` every 800ms for real-time updates.

**Progress stages mapped to percentage:**
| Stage | Percent | What happens |
|-------|---------|-------------|
| Motion scan | 0–18% | Farneback optical flow on every 3rd frame |
| Frame extraction | 18–25% | Parallel extraction of surviving keyframes (4 threads) |
| CLIP embedding | 25–85% | Batched GPU inference, updates per batch |
| Saving | 85–100% | Thumbnails written + index.json serialized |

Each update includes `stage`, `percent`, `message` (human-readable stage description), and `eta_seconds` (estimated time remaining). The frontend displays an animated gradient progress bar with this data.

### 5.6 Persistence

After indexing, the full `VideoIndex` dataclass (frame_indices, timestamps, embeddings, motion_scores, total_frames) is serialized to `storage/frames/{video_id}/index.json`. On server startup, the lifespan handler reloads all saved indexes from disk.

---

## 6. API Reference

All endpoints are prefixed with `/api`. The frontend Vite dev server proxies `/api` to `localhost:8000`.

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/api/health` | Server status | — | `{status, indexed_videos}` |
| POST | `/api/videos/upload` | Upload video | multipart file | `{video_id, filename, size}` |
| POST | `/api/videos/{id}/index` | Start async indexing | — | `{video_id, status: "indexing_started"}` |
| GET | `/api/videos/{id}/index-progress` | Poll indexing progress | — | `{stage, percent, message, eta_seconds, ...}` |
| GET | `/api/videos/{id}/status` | Index readiness | — | `{video_id, keyframes, timestamps, status}` |
| POST | `/api/search` | Semantic search | `{query, top_k}` | `{video_id, segments[], query_info, status}` |
| GET | `/api/clips/{id}` | Download clip | query: `start_frame, end_frame` | MP4 file |
| GET | `/api/reports/{id}` | Generate report | — | `{video_id, keyframes_count, frame_reduction_pct, ...}` |
| GET | `/api/metrics` | Dashboard metrics | — | `{videos_indexed, total_keyframes, videos[]}` |
| GET | `/api/frames/{id}/frame_{n}.jpg` | Thumbnail | — | JPEG image |

---

## 7. Performance Optimizations

### Implemented Optimizations

| Area | Before | After | Speedup |
|------|--------|-------|---------|
| Motion scan resolution | 320×180 (57,600 px) | 160×90 (14,400 px) | 4× fewer pixels |
| Flow computation | Every frame | Every 3rd frame (stride=3) | 66% fewer computations |
| Frame selection | Fixed threshold (0.3) | Adaptive percentile (top 5%) | No manual tuning |
| Frame extraction | Single-threaded sequential | 4 threads parallel | 3-4× faster |
| CLIP inference | Unbatched | Batch size 32 | Up to 32× throughput |
| Vector search | Brute force (IndexFlatIP) | Approximate (IndexIVFFlat) | O(n) → O(log n) |
| Thumbnail saving | Sequential loop | ThreadPoolExecutor.map | 4× faster |

### Bottleneck Analysis

| Stage | 5s video (90 frames) | 8min video (14,400 frames) | Scaling |
|-------|---------------------|---------------------------|---------|
| Motion scan | < 1s | ~30-60s | Linear with frame count |
| Frame extraction | < 1s | ~10-20s | Linear with keyframes |
| CLIP embedding | ~13s | ~2-5 min | Linear with keyframes |
| **Total** | **~13s** | **~3-6 min** | — |

The CLIP embedding stage is the dominant bottleneck for long videos. It scales linearly with the number of keyframes (~100ms per 32-frame batch on GPU).

---

## 8. Project Structure

```
AUTOMATE/
├── .gitignore                # Excludes videos, node_modules, __pycache__, storage, venv
├── backend/
│   ├── main.py              # FastAPI server (10 endpoints + lifespan handler + async indexing)
│   ├── pipeline.py          # CV pipeline (motion sampling, CLIP, FAISS, extraction, progress)
│   └── storage/             # Local data storage
│       ├── originals/       # Uploaded video files
│       ├── frames/          # Keyframe thumbnails + index.json per video
│       ├── clips/           # Extracted MP4 clips
│       └── reports/         # Generated JSON reports
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component (upload + search + progress bar + results)
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Tailwind imports
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
├── test_workflow.ps1        # End-to-end API test script (with progress polling)
└── README.md                # Quick-start guide
```

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `backend/pipeline.py` | ~253 | Core CV pipeline: motion sample, embedding, search, clip extraction, progress tracking |
| `backend/main.py` | ~180 | FastAPI application: routes, middleware, lifespan, thread pool, async index, progress endpoint |
| `frontend/src/App.jsx` | ~218 | React SPA: upload, search, results, progress bar with ETA, polling, activity log |
| `test_workflow.ps1` | ~100 | Automated API test: health → upload → index (with progress polling) → search → report |

---

## 9. Metrics

| Metric | Measured Value | Method |
|--------|---------------|--------|
| Frame reduction (5s video) | 53-63% | Adaptive percentile motion sampling |
| Frame reduction (8min video) | ~97% | Same — longer video has more static frames |
| Motion scan speed | ~30s for 14K frames | 160×90 gray, stride=3, 200+ FPS effective |
| CLIP embedding throughput | ~300 frames/min | Batch size 32 on GPU |
| Query latency | < 5s | CLIP text embed + FAISS IVFFlat search |
| Format support | mp4, avi, mov, mkv, webm | OpenCV VideoCapture codec support |
| No-match rejection | Confidence-gated | Thresholds: high > 0.25, medium > 0.15 |
| Index persistence | Survives restarts | JSON → lifespan handler reload |
| Max video tested | 8 min (177 MB, 4K) | 14,400 frames, ~400 keyframes retained |

---

## 10. Future Scope

| Feature | Effort | Impact |
|---------|--------|--------|
| Open-vocabulary detection (Grounding DINO) | High | Unlimited object classes |
| Multi-object tracking (ByteTrack) | High | Eliminates single-frame false positives |
| Action recognition (X-CLIP) | Medium | Event-level understanding |
| Speech-to-text search (Whisper) | Low | Search by spoken content |
| Privacy blur (face detection) | Low | Responsible AI compliance |
| Cross-camera search | Medium | Multi-camera investigation |
| Real-time CCTV integration | High | Live video search |
| GPU FAISS (index on GPU) | Low | Faster search at 100K+ vectors |
| ONNX runtime for CLIP | Medium | Faster inference, no PyTorch dep |
| HNSW index instead of IVF | Low | Better accuracy/speed tradeoff |

---

*Documentation updated to reflect the implemented MVP. Originally conceived as a hackathon project for Team Return0.*
