<div align="center">

# 🎬 SceneTrace AI

### *"Show me the person in a red jacket near the entrance."* — and it finds it.

**Natural Language Video Grounding** · Hackathon Submission · Problem Statement #5 · [Team Return0]

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-FF6F00?logo=openai&logoColor=white)](https://openai.com/research/clip)
[![YOLO-World](https://img.shields.io/badge/YOLO--World-Large-00C853?logo=ultralytics&logoColor=white)](https://docs.ultralytics.com/models/yolo-world)
[![FAISS](https://img.shields.io/badge/FAISS-IVFFlat-512BD4?logo=meta&logoColor=white)](https://faiss.ai)
[![Grounding DINO](https://img.shields.io/badge/GroundingDINO-ZeroShot-22c55e?logo=huggingface&logoColor=white)](https://huggingface.co/IDEA-Research/grounding-dino-base)
[![GPU](https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)
[![Tests](https://img.shields.io/badge/Tests-37_passing-22c55e?logo=pytest)](https://github.com/NYN-05/SceneTrace_AI)
[![Security](https://img.shields.io/badge/Security-CORS_restricted-6366f1)](https://github.com/NYN-05/SceneTrace_AI)

**Upload a video. Type a sentence. Get the clip.** — All on-device, no cloud API, ~5s query time.

</div>

---

## 🔥 The Problem

> *"Input a text description to semantically search and retrieve matching clips from recorded video archives using NLP and vision-language models."*

**The old way:** Watch 8 hours of footage manually. Or use motion sensors that can't tell a person dropping a backpack from a leaf blowing in the wind.

**SceneTrace AI:** Type *"a person picking up a backpack near the side door"* → get the exact 12-second clip with 91% confidence, timestamped and previewed.

---

## ⚡ Quick Start

```powershell
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
python main.py

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** → Upload `.mp4` → Type any query → Get results.

> Optional: copy `backend/.env.example` to `backend/.env` and customize.

---

## 🧠 How It Works

```
                    INDEXING                           ENHANCED SEARCH
┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐   ┌──────────────────────┐
│  Upload  │──▶│  Motion      │──▶│  CLIP    │──▶│  FAISS     │──▶│  Zero-Shot           │
│  Video   │   │  Sampling    │   │  Embed   │   │  Search    │   │  Detection           │
│          │   │  (160×90,    │   │  (GPU    │   │  (IVF      │   │  (YOLO-World-L,       │
│          │   │   stride=3)  │   │  batch32)│   │  FlatIP)   │   │   auto fallback      │
│          │   │              │   │          │   │            │   │   M → S)             │
└──────────┘   └──────┬───────┘   └──────────┘   └─────┬──────┘   └───────┬──────────────┘
                      │                                 │                 │
                      ▼                                 ▼                 ▼
               ┌────────────┐                    ┌──────────┐     ┌──────────────┐
               │ Background │                    │ Weighted │     │  Bounding    │
               │ ThreadPool │                    │ Scoring  │◀────│  Boxes +     │
               │ Executor   │                    │ (CLIP +  │     │  Labels      │
               └────────────┘                    │ Object)  │     └──────────────┘
                                                 └─────┬────┘
                                                       ▼
                                               ┌──────────────┐
                                               │  Explanation │
                                               │  + Timeline  │
                                               │  + Dashboard │
                                               └──────────────┘
```

### 1️⃣ Motion-Guided Sampling
Frame differencing (default) or Farneback optical flow at **160×90** resolution, every **3rd frame**, adaptive percentile threshold. Keeps only the **~5% most active frames** — **up to 97% reduction** on surveillance footage.

### 2️⃣ CLIP Embeddings
Each keyframe → 512-dim vector via `openai/clip-vit-base-patch32`. The user's query goes through the same text encoder. Both live in the **same semantic space**.

### 3️⃣ FAISS Vector Search
Milliseconds to search. IVFFlat for large indexes, FlatIP fallback for small ones. Results clustered into coherent segments by frame-index proximity.

### ⏳ Live Progress Tracking
Indexing runs **asynchronously in a background thread** — the UI polls `GET /api/videos/{id}/index-progress` every 800ms.

### 4️⃣ Zero-Shot Object Detection
**YOLO-World-L** (default) runs open-vocabulary detection on candidate frames. If inference fails, the system automatically falls back to **YOLO-World-M → YOLO-World-S** without user intervention. Can also use **Grounding DINO** via `DETECTOR_BACKEND=grounding_dino`.

### 5️⃣ Weighted Scoring + Explanation
**Overall Score = 55% Semantic Similarity + 45% Object Match**
Every result shows a score breakdown with an **Explanation Panel**.

### 6️⃣ Confidence Gating
| Level | Threshold | Behavior |
|-------|-----------|----------|
| 🟢 HIGH | > 0.25 | Strong semantic match |
| 🟡 MEDIUM | > 0.15 | Plausible match, review suggested |
| 🔴 LOW | < 0.15 | Likely irrelevant, shown last |

---

## 📊 Benchmarks

| Metric | Value |
|--------|-------|
| **5s video index** (90 frames) | **13.4 seconds** |
| **8min video index** (14,400 frames) | **3–6 minutes** |
| **Query latency** | **< 5 seconds** |
| **Frame reduction** (short video) | **62.7%** |
| **Frame reduction** (long video) | **~97%** |
| **Hardware** | Consumer laptop + CUDA GPU |
| **CLIP model size** | ~600 MB (loaded once) |

> ⏱️ *First index is slower — CLIP model loads into GPU memory on first request.*

---

## 🖥️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Vision-Language** | CLIP ViT-B/32 | Zero-shot semantic understanding |
| **Open-Vocab Detection** | YOLO-World-L (→M→S fallback) + Grounding DINO | Detect any object from query text |
| **Motion Analysis** | OpenCV Farneback (160×90) | 200+ FPS flow computation |
| **Vector Search** | FAISS (IVFFlat) | O(log n) at 100K+ vectors |
| **Backend** | FastAPI + ThreadPoolExecutor | Async endpoints, CPU offload, metrics |
| **Frontend** | React 18 + Vite + Tailwind | Google-like search, dashboard, timeline |
| **Storage** | Local filesystem (JSON + FAISS binary) | Index persistence across restarts |

---

## 📡 API Reference

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `GET` | `/api/health` | 🩺 Server status + indexed count |
| `POST` | `/api/videos/upload` | 📤 Upload `.mp4` / `.avi` / `.mov` |
| `POST` | `/api/videos/{id}/index` | ⚙️ Index with live progress tracking |
| `GET` | `/api/videos/{id}/index-progress` | 📊 Pollable progress % + ETA + stage |
| `GET` | `/api/videos/{id}/status` | ✅ Index readiness + keyframe count |
| `POST` | `/api/search` | 🔍 Original semantic search (backward compat) |
| `POST` | `/api/v2/search` | 🚀 Enhanced search with detection + weighted scoring + score breakdown |
| `GET` | `/api/videos/{id}/timeline` | ⏳ Event timeline with motion scores per frame |
| `GET` | `/api/videos/{id}/objects` | 🔲 Detected objects with annotated thumbnails |
| `POST` | `/api/search/suggest` | 💡 Query autocomplete suggestions |
| `GET` | `/api/dashboard/metrics` | 📊 Full performance dashboard (speed, latency, GPU, reduction) |
| `GET` | `/api/clips/{id}` | 🎞️ Download extracted MP4 clip |
| `GET` | `/api/reports/{id}` | 📈 Frame reduction + motion report |
| `GET` | `/api/metrics` | 📊 Basic metrics |

---

## 📁 Project Structure

```
├── .gitignore
├── backend/
│   ├── main.py              # FastAPI server (16 endpoints + security + async indexing)
│   ├── pipeline.py          # CV pipeline (motion, CLIP, FAISS, extraction, progress)
│   ├── search_engine.py     # Enhanced search: detection + weighted scoring + suggestions
│   ├── detector.py          # YOLO-World (L→M→S fallback) + Grounding DINO (lazy-loaded)
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
│   └── storage/             # originals/, frames/, clips/, reports/
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main SPA (slim — imports components)
│   │   ├── main.jsx         # React entry point
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
│   │       └── useApi.js    # Custom hooks for search + upload logic
│   └── package.json
├── Docs/
│   ├── TECHNICAL_BRIEF.md   # Full project justification
│   └── SceneTrace_AI_Final_Idea.md
├── test_workflow.ps1        # 12-step automated test suite
├── stop_servers.ps1         # Kill servers on ports 8000 & 5173
└── README.md
```

---

## 🧪 Running Tests

```powershell
# Unit + API tests (no server needed) — from root
python -m pytest backend\tests\ -v

# Or from the backend directory
cd backend; python -m pytest tests/ -v

# End-to-end integration (servers must be running)
.\test_workflow.ps1 -VideoPath test.mp4
```

**37 pytest tests** covering: pipeline logic (segments, queries, FAISS, embeddings, motion), search engine (suggestions), benchmark singleton, config/env, and 12 FastAPI endpoint tests (health, upload validation, input sanitization, error handling).

End-to-end `test_workflow.ps1` validates: health → upload → index (progress polling) → status → search(v1) → report → **v2 search** → **suggestions** → **dashboard metrics** → **timeline** → **objects**

---

## 🏆 Why It Wins

- **✅ YOLO-World with auto fallback** — YOLO-World-L (default) with seamless fallback to Medium → Small on failure. Zero-shot object detection on any query term. Grounding DINO also available.
- **✅ Explainable AI** — Every result shows a score breakdown (semantic, object, tracking, temporal) with a "Why this matched" explanation panel.
- **✅ Google-like UX** — Rich search cards, autocomplete suggestions, score breakdown bars, dashboard metrics, event timeline.
- **✅ Performance dashboard** — Real-time metrics: indexing speed (fps), avg query latency, frame reduction %, GPU status.
- **✅ Fully working end-to-end** — Not a prototype. Upload any `.mp4`, type any query, get results with bounding boxes.
- **✅ Live demo** — Running on this machine at `localhost:5173`. Judges can test it in 30 seconds.
- **✅ No cloud dependency** — All on-device (CLIP, FAISS, YOLO-World, FastAPI). Private, free, offline-capable.
- **✅ Semantic understanding** — CLIP matches by concept, not keyword. "Red jacket" works in any lighting, any angle.
- **✅ Optimized for real footage** — 97% frame reduction, ~5s query time, async indexing with progress bar.

---

<div align="center">

**Built by Team Return0** · Submission for Problem Statement #5

[![GitHub](https://img.shields.io/badge/GitHub-SceneTrace__AI-181717?logo=github&logoColor=white)](https://github.com/NYN-05/SceneTrace_AI)

</div>
