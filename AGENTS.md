# SceneTrace AI — Agent Guide

## Commands

```powershell
# Run all tests
python -m pytest backend/tests/ -v

# Run single test
python -m pytest backend/tests/test_search.py::test_suggest_car -v

# Start dev server (no --reload to avoid reloader crashing silently)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Run full workflow test against T2.mp4
.\test_workflow.ps1 -VideoPath T2.mp4
```

## Architecture

- **`backend/main.py`** — FastAPI app (routes, upload, index orchestration)
- **`backend/pipeline.py`** — Core indexing pipeline (motion scan, CLIP embed, object detection/tracking, captioning, FAISS index)
- **`backend/search_engine.py`** — Multi-signal hybrid search (CLIP, caption, object, motion, track, temporal, relationship)
- **`backend/config.py`** — Settings from `.env` + BenchmarkStats singleton
- **`backend/detector.py`** — YOLO-World object detection wrapper

## Known Bugs & Fixes Applied

| Bug | Fix | File |
|-----|-----|------|
| Clip embeddings are 515-dim (512 CLIP + 3 motion); text query is 512-dim → dot product crashes | Trim `clip_embs[:, :512]` when shape[1]==515 and q_emb shape[1]==512 | `search_engine.py:337-339` |
| `Captioner._load()` fails on missing `einops`, retries every frame → log spam | Cache failure via `self._model = False` in `caption()` | `pipeline.py:285-293` |
| `MetadataDB` dir didn't exist before `sqlite3.connect()` | `mdb_dir.mkdir(parents=True, exist_ok=True)` before creating DB | `pipeline.py:739-740` |
| `TRACK_CONSISTENCY_WEIGHT` defined twice in `config.py` and `.env` | Removed duplicate at `config.py:63` and `.env:50` | `config.py`, `.env` |

## Conventions

- Run `pytest` after any code change (50 tests, ~13s).
- Server process crashes silently when `--reload` parent dies. Use `--reload` only during dev, remove for production.
- `.env` enables captioner (`CAPTIONER_ENABLED=true`). Reranker (`RERANKER_ENABLED=false`) is disabled because `BAAI/bge-reranker-v2-m3` model isn't downloaded.
- Florence-2 model cached at `~/.cache/huggingface/hub/models--microsoft--Florence-2-base/`. If missing, run `python backend/download_model.py`.
- `test_workflow.ps1` uses `urllib.request` (stdlib), not `httpx`.
- The PowerShell upload helper is embedded inline in `test_workflow.ps1` — it generates a temp `upload.py`.
- Index progress is available via SSE at `GET /api/videos/{video_id}/index-progress/stream` (single connection, no polling). The legacy polling endpoint is kept for backward compatibility.
