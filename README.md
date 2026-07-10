<div align="center">

# 🎬 SceneTrace AI

### *"Show me the person in a red jacket near the entrance."* — and it finds it.

**Natural Language Video Grounding** · Hackathon Submission · Problem Statement #5

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B/32-FF6F00?logo=openai&logoColor=white)](https://openai.com/research/clip)
[![FAISS](https://img.shields.io/badge/FAISS-IVFFlat-512BD4?logo=meta&logoColor=white)](https://faiss.ai)
[![GPU](https://img.shields.io/badge/GPU-CUDA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda)

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
cd backend; python main.py

# Terminal 2 — Frontend
cd frontend; npm install; npm run dev
```

Open **http://localhost:5173** → Upload `.mp4` → Type any query → Get results.

---

## 🧠 How It Works

```
                INDEXING                                  SEARCH
┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Upload  │──▶│  Motion      │──▶│  CLIP    │──▶│  FAISS   │──▶│  Segment     │
│  Video   │   │  Sampling    │   │  Embed   │   │  Search  │   │  Clustering  │
│          │   │  (160×90,    │   │  (GPU    │   │  (IVF    │   │  + Confidence│
│          │   │   stride=3)  │   │  batch32)│   │  FlatIP) │   │  Gating      │
└──────────┘   └──────────────┘   └──────────┘   └──────────┘   └──────────────┘
```

### 1️⃣ Motion-Guided Sampling
Farneback optical flow at **160×90** resolution, every **3rd frame**, adaptive percentile threshold. Keeps only the **~5% most active frames** — **up to 97% reduction** on surveillance footage.

### 2️⃣ CLIP Embeddings
Each keyframe → 512-dim vector via `openai/clip-vit-base-patch32`. The user's query goes through the same text encoder. Both live in the **same semantic space** — understands concepts, not keywords.

### 3️⃣ FAISS Vector Search
Milliseconds to search. IVFFlat for large indexes, FlatIP fallback for small ones. Results clustered into coherent segments by frame-index proximity.

### 4️⃣ Confidence Gating
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
| **Motion Analysis** | OpenCV Farneback (160×90) | 200+ FPS flow computation |
| **Vector Search** | FAISS (IVFFlat) | O(log n) at 100K+ vectors |
| **Backend** | FastAPI + ThreadPoolExecutor | Async endpoints, CPU offload |
| **Frontend** | React 18 + Vite + Tailwind | Real-time progress bar, segment cards |
| **Storage** | Local filesystem (JSON) | Index persistence across restarts |

---

## 📡 API Reference

| Method | Endpoint | What it does |
|--------|----------|-------------|
| `GET` | `/api/health` | 🩺 Server status + indexed count |
| `POST` | `/api/videos/upload` | 📤 Upload `.mp4` / `.avi` / `.mov` |
| `POST` | `/api/videos/{id}/index` | ⚙️ Index with live progress tracking |
| `GET` | `/api/videos/{id}/index-progress` | 📊 Pollable progress % + ETA + stage |
| `GET` | `/api/videos/{id}/status` | ✅ Index readiness + keyframe count |
| `POST` | `/api/search` | 🔍 Semantic search by natural language |
| `GET` | `/api/clips/{id}` | 🎞️ Download extracted MP4 clip |
| `GET` | `/api/reports/{id}` | 📈 Frame reduction + motion report |
| `GET` | `/api/metrics` | 📊 Dashboard metrics |

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server (9 endpoints)
│   ├── pipeline.py          # CV pipeline (motion, CLIP, FAISS)
│   └── storage/             # originals/, frames/, clips/, reports/
├── frontend/
│   └── src/App.jsx          # React dashboard with progress bar
├── Docs/
│   ├── TECHNICAL_BRIEF.md   # Full project justification
│   └── SceneTrace_AI_Final_Idea.md
├── test_workflow.ps1        # 7-step automated test suite
├── stop_servers.ps1         # Kill servers on ports 8000 & 5173
└── README.md
```

---

## 🧪 Running Tests

```powershell
.\test_workflow.ps1 -VideoPath test.mp4
```

Validates: health → upload → index (with progress polling) → status → search → report → metrics.

---

## 🏆 Why It Wins

- **✅ Fully working end-to-end** — Not a prototype. Upload any `.mp4`, type any query, get results.
- **✅ Live demo** — Running on this machine at `localhost:5173`. Judges can test it in 30 seconds.
- **✅ No cloud dependency** — All on-device (CLIP, FAISS, FastAPI). Private, free, offline-capable.
- **✅ Semantic understanding** — CLIP matches by concept, not keyword. "Red jacket" works in any lighting, any angle.
- **✅ Optimized for real footage** — 97% frame reduction, ~5s query time, handles 4K video.
- **✅ Measured results** — Every claim backed by wall-clock timing, verified by automated test suite.

---

<div align="center">

**Built by Team Return0** · Submission for Problem Statement #5

[![GitHub](https://img.shields.io/badge/GitHub-SceneTrace__AI-181717?logo=github&logoColor=white)](https://github.com/NYN-05/SceneTrace_AI)

</div>
