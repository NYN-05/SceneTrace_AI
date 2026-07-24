# SceneTrace AI — Agent Guide

## Commands

```powershell
# All 77 tests (no server needed, run from project root)
python -m pytest backend/tests/ -v

# Single test
python -m pytest backend/tests/test_search.py::TestExtractClassNames::test_single_class -v

# Dev servers (two terminals)
cd backend; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
cd frontend; npm install && npm run dev

# Full E2E workflow against a running backend
python backend/run_workflow.py "path\to\video.mp4" "your query"
```

## Key Files

| File | Role |
|------|------|
| `backend/main.py` | FastAPI app (16 endpoints, upload, async index orchestration, SSE progress stream) |
| `backend/pipeline.py` | Indexing pipeline: motion scan (diff/Farneback at 160×90, stride=3), CLIP embedding, object detection/tracking (SimpleTracker IoU), optional captioning (Florence-2), FAISS build |
| `backend/search_engine.py` | 4-stage hybrid search: FAISS semantic → SQLite class filter → 6-signal weighted scoring → optional cross-encoder reranker |
| `backend/config.py` | `Settings` from `.env` + `BenchmarkStats` singleton. `load_dotenv(dotenv_path=Path(__file__).parent / ".env")` |
| `backend/detector.py` | YOLO-World (L→M→S auto fallback from `YOLO_WORLD_FALLBACKS` env var) or Grounding DINO. YOLO calls serialized under `_yolo_bool_lock` |
| `backend/tests/` | 6 test modules (pytest), no external services needed |

## Architecture Gotchas

- **Env loading**: `config.py` loads `backend/.env` via explicit script-dir path. `load_dotenv` does NOT override existing env vars. Running from wrong CWD used to silently miss `.env` — this is fixed.
- **No `--reload`**: uvicorn reloader spawns a child that dies silently on background thread errors. Start without `--reload` for production.
- **Indexing is async**: `index_video` runs in `executor.submit()` background thread. Progress shared via `_index_progress` dict under `_index_progress_lock`.
- **FAISS lazy-loaded**: `.faiss` files read on demand per-video, cached in `_faiss_index_cache` under `_faiss_cache_lock`. Not all held in RAM.
- **Clip embeddings 515-dim**: 512 CLIP + 3 motion stats. Text queries are 512-dim. `_search_clip_embeddings` trims to `[:, :512].copy()` before dot product.
- **Scoring weights**: 6 signals sum to 1.10 (intentionally non-normalized). Scores clamped to [0,1] at every FAISS boundary.
- **Search modifies no shared state**: `_search_objects_faiss` uses a local `obj_meta` slice instead of mutating `idx.object_metadata`.

## Subtle Bugs (Don't Reintroduce)

| Problem | Solution |
|---------|----------|
| CLIP `compute_embeddings` leaks preproc future on exception path | `try`/`except` that calls `submitted.result()` before `raise` |
| FAISS returns `-1` indices / `-inf` scores when `k > n` or NaN in embeddings | `np.nan_to_num` + `math.isfinite` + clamp to [0,1] at every search boundary |
| FAISS gap detection used similarity-rank order instead of temporal order → merged non-adjacent frames | Sort FAISS results by `idx.frame_indices[x[0]]` before `frames_to_segments` |
| Clip embedding view `[:, :512]` is non-contiguous → slow numpy ops | `.copy()` after slicing |
| `object_metadata` length mismatches `object_embs` (from zero-area crops) | Truncate to min length with warning instead of returning empty — but use local slice, don't mutate the shared `VideoIndex` |
| YOLO-World `torch.Tensor.__bool__` ambiguity on multi-element tensors | Monkey-patch under `_yolo_bool_lock` around `model(image)`, restore original in `finally` |
| `Captioner._load()` fails once → retries every frame → log spam | Cache failure via `self._model = False` so `caption()` skips silently |
| Frame indices can exceed `total - 1` during uniform frame insertion | Clamp to `max(total - 1, 0)` and filter with `0 <= f <= max_valid` |
| `stride` loop in `compute_embeddings` sets `preproc_future = None` only in `else` branch → wrong ordering | Reset `preproc_future = None` explicitly after consuming the future, use separate `submitted` variable |

## Frontend Quirks

- **Score breakdown keys must match backend**: `clip_semantic`, `caption_similarity`, `object_match`, `motion_activity`, `tracking_consistency`, `temporal_alignment`, `relationship_overlap`. The old `semantic_similarity` / `temporal_match` keys render zeros.
- **Polling interval leaked on unmount**: `useUpload` hook must clean up `setInterval` via `useEffect` return.
- **`navigator.clipboard.writeText`** returns a promise — must be `await`ed with try/catch.

## Dependencies

- `slowapi` and `fastapi-limiter` were removed — they were imported but never applied to any endpoint.
- `python-magic` is optional (try/except guarded), used for MIME validation on upload.
- `sentence-transformers` only needed if `RERANKER_ENABLED=true` (disabled by default).
- Florence-2 (~800MB) only needed if `CAPTIONER_ENABLED=true`: run `python backend/download_model.py`.

## Test Quirks

- Run from project root with `python -m pytest backend/tests/ -v`. Don't `cd backend` then run `pytest` — the `load_dotenv` path is relative to `config.py`, not CWD.
- `pytest.ini` suppresses third-party warnings (faiss SwigPy deprecations, starlette testclient, Windows asyncio overlapped).
- `test_config.py::test_env_override` uses `importlib.reload(config)` — the logger handler guard (`if not _root_logger.handlers`) prevents duplicate handler accumulation on reload.
- `test_api.py` uses `fastapi.testclient.TestClient`. The `StarletteDeprecationWarning` for httpx is suppressed in pytest.ini.
