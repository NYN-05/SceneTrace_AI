# SceneTrace AI — Agent Guide

## Commands

```powershell
# Tests (run from project root, NOT `cd backend`)
python -m pytest backend/tests/ -v                          # all 77
python -m pytest backend/tests/test_search.py::TestExtractClassNames::test_single_class -v

# Lint
ruff check backend\                                         # default rules
ruff check backend\ --select ALL                            # +cosmetic (expect 1136)

# Dev servers
cd backend; uvicorn main:app --host 0.0.0.0 --port 8000    # no --reload
cd frontend; npm run dev                                    # Vite on :5173

# E2E smoke test (backend must be running)
python backend/run_workflow.py "path\to\video.mp4" "your query"
```

`.env` lives at `backend/.env` — loaded by `config.py` via `Path(__file__).parent / ".env"`. `load_dotenv()` does **not** override existing env vars. Run tests from project root, not `backend/`, so CWD-relative `.env` paths don't interfere.

No `--reload`: uvicorn reloader spawns a child that dies silently on background thread errors. Use two terminals.

## Entrypoints

| File | Role |
|------|------|
| `backend/main.py` | FastAPI app: 16 endpoints, SSE progress stream, dual thread pools |
| `backend/pipeline.py` | Indexing: motion scan → CLIP embed → detect/track → caption → FAISS build (all parallel) |
| `backend/search_engine.py` | 4-stage hybrid search: FAISS semantic → SQLite class filter → 6-signal scoring → optional cross-encoder reranker |
| `backend/config.py` | `Settings` from `os.getenv` + `BenchmarkStats` singleton |
| `backend/detector.py` | YOLO-World (L→M→S fallback) or Grounding DINO |

No `__init__.py` in `backend/` (implicit namespace package). No `pyproject.toml`, no pre-commit, no CI. Tests are plain pytest with `pytest.ini`.

## Static Analysis

Only `ruff` is available (no mypy/pyright/flake8). Default `ruff check backend\` must be clean before committing. The `--select ALL` level adds ~107 largely cosmetic issues (type hints, docstrings, line length, asserts in tests) — skip those.

## Architecture

- **Parallelism**: All heavy work is CPU-threaded via `ThreadPoolExecutor` (not asyncio). Only the FastAPI handlers use asyncio. `_index_executor` (2 workers), `_search_executor` (8 workers). `_index_semaphore(2)` caps concurrent indexing.
- **Indexing pipeline runs 3 blocks in parallel** via a 2-worker `capdet_pool`: captioning and detection run concurrently; FAISS index builds (main + object + caption) run concurrently via `ThreadPoolExecutor.map`.
- **Search runs 4 sub-searches in parallel**: `_run_sem`, `_run_obj`, `_run_cap`, `_run_clip` via `ThreadPoolExecutor(max_workers=4)`. Per-segment scoring also parallel when >4 candidates.
- **FAISS lazy-loaded**: `.faiss` files read on demand per-video, cached in `_faiss_index_cache` under `_faiss_cache_lock`, trimmed to 30 max.
- **Clip embeddings are 515-dim**: 512 CLIP + 3 motion stats. Text queries are 512-dim. `_search_clip_embeddings` trims `[:, :512].copy()` before dot product.
- **Scoring weights**: 6 signals sum to 1.10 (intentionally non-normalized). Scores clamped to [0,1] at every FAISS boundary.
- **`q_emb` computed once**: `_get_query_embedding()` at `search_engine.py:509` with LRU cache (max 64, thread-safe). Passed as `q_emb=` kwarg to all 4 subroutines.
- **`_SQLITE_CONN_CACHE`** keyed by `vid` with health-check reconnect, reused across searches.

## Frontend Quirks

- **SSE replaces polling**: `useUpload` connects `EventSource` to `/index-progress/stream`. Cleanup on unmount via `useEffect` return. Heartbeat every 15s keeps proxy alive.
- **Score breakdown keys must match backend**: `clip_semantic`, `caption_similarity`, `object_match`, `motion_activity`, `tracking_consistency`, `temporal_alignment`, `relationship_overlap`. Old keys render zeros.
- **`navigator.clipboard.writeText`** returns a promise — must be `await`ed with try/catch.
- **AbortController per search**: `doSearch` creates new `AbortController`, aborts previous in-flight fetch. `handleInputChange` debounced at 250ms.
- **Log capped at 50**: `[...p.slice(-49), msg]` + `LogItem` memo component prevents full-list re-render.
- **Bundle splitting**: `DashboardTab` and `TimelineTab` loaded via `React.lazy(() => import(...))` + `Suspense`.

## Subtle Bugs (Don't Reintroduce)

| Problem | Solution |
|---------|----------|
| CLIP `compute_embeddings` leaks preproc future on exception path | `try`/`except` that calls `submitted.result()` before `raise` |
| FAISS returns `-1` indices / `-inf` scores when `k > n` or NaN in embeddings | `np.nan_to_num` + `math.isfinite` + clamp to [0,1] at every search boundary |
| FAISS gap detection used similarity-rank order instead of temporal order → merged non-adjacent frames | Sort FAISS results by `idx.frame_indices[x[0]]` before `frames_to_segments` |
| Clip embedding view `[:, :512]` is non-contiguous → slow numpy ops | `.copy()` after slicing |
| `object_metadata` length mismatches `object_embs` (from zero-area crops) | Truncate to min length with warning — use local slice, don't mutate shared `VideoIndex` |
| YOLO-World `torch.Tensor.__bool__` ambiguity on multi-element tensors | Monkey-patch under `_yolo_bool_lock` around `model(image)`, restore original in `finally` |
| `Captioner._load()` fails once → retries every frame → log spam | Cache failure via `self._model = False` so `caption()` skips silently |
| Frame indices can exceed `total - 1` during uniform frame insertion | Clamp to `max(total - 1, 0)` and filter with `0 <= f <= max_valid` |

## Test Quirks

- Run from project root with `python -m pytest backend/tests/ -v`. Don't `cd backend` then `pytest` — CWD-relative `.env` load breaks.
- `pytest.ini` suppresses faiss SwigPy deprecations, starlette testclient, Windows asyncio overlapped warnings.
- `test_config.py::test_env_override` uses `importlib.reload(config)` — logger handler guard (`if not _root_logger.handlers`) prevents duplicate handlers on reload.
- 77 tests, no external services, no GPU required. FastAPI test client for API tests.
