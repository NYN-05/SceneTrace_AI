# SceneTrace AI

Natural language video grounding — describe an event in plain English, get the relevant clip with evidence.

## Project Structure

```
├── backend/
│   ├── main.py          # FastAPI server (8 endpoints)
│   ├── pipeline.py      # CV pipeline: motion sampling, CLIP, FAISS search
│   └── storage/         # originals/, frames/, clips/, reports/
├── frontend/
│   └── src/App.jsx      # React dashboard (upload + search)
├── stop_servers.ps1     # Kill processes on ports 8000 and 5173
├── test_workflow.ps1    # End-to-end API test script
└── README.md
```

## Prerequisites

- Python 3.10+ with packages: torch, opencv-python, transformers, faiss-cpu, fastapi, uvicorn, numpy, Pillow, python-multipart, httpx
- Node.js 18+ with npm

## How to Use

### 1. Start the Backend

Open a terminal and run:
```powershell
cd backend
python main.py
```
Server starts at `http://localhost:8000`. Keep this terminal open.

### 2. Start the Frontend

Open a second terminal and run:
```powershell
cd frontend
npm install
npm run dev
```
App opens at `http://localhost:5173`. Keep this terminal open.

### 3. Use the UI

Open `http://localhost:5173` in a browser.

| Tab | What to do | Expected result |
|-----|-----------|----------------|
| **Upload** | Click file picker → select `.mp4` → click **Upload & Index** | Status shows: Uploading → Indexing... → Ready (N keyframes) |
| **Search** | Type a query like `"a red circle moving"` → press Enter or click **Search** | Confidence badge (HIGH/MEDIUM/LOW) + segment cards with thumbnails and match % |

### 4. Stop Servers

```powershell
.\stop_servers.ps1
```

### 5. Test via Command Line (servers must be running)

```powershell
.\test_workflow.ps1 -VideoPath test.mp4
```

## Time Estimates

| Video length | Step | Time |
|-------------|------|------|
| 5 seconds (90 frames) | Motion sampling | < 1s |
| | CLIP embedding | ~13s |
| | **Total index** | **~13s** |
| 8 minutes (14,400 frames) | Motion sampling | ~30-60s |
| | CLIP embedding | ~2-5 min |
| | **Total index** | **~3-6 min** |
| | Search query | **< 5s** |

First index is slowest — CLIP model (~600MB) loads into GPU memory on first use.

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Server status |
| POST | `/api/videos/upload` | Upload video |
| POST | `/api/videos/{id}/index` | Index video |
| GET | `/api/videos/{id}/status` | Index status |
| POST | `/api/search` | Natural language search |
| GET | `/api/clips/{id}` | Extract clip by frame range |
| GET | `/api/reports/{id}` | Index summary report |
| GET | `/api/metrics` | Dashboard metrics |

## Pipeline

1. **Motion-guided sparse sampling** — Farneback optical flow skips static frames (50-90% reduction)
2. **CLIP embeddings** — 512-dim vectors per keyframe via `openai/clip-vit-base-patch32`
3. **FAISS search** — Cosine similarity search with segment clustering
4. **Confidence gating** — Results classified as high/medium/low
