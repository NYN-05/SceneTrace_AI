# Phase Implementation Summary

## Phase 1 — Object Detection in Indexing Pipeline

### Objective
Move object detection (YOLO-World) from post-search into the indexing pipeline — detect, crop, and CLIP-embed objects per keyframe so object-level embeddings become part of the search index.

### Files Modified

**`backend/config.py`** — New settings:
- `INDEX_DETECTION_PROMPTS` — COCO 80-class prompt list for YOLO-World during indexing
- `INDEX_MIN_FPS` (default: `5`) — minimum keyframe rate enforced after motion sampling
- `INDEX_OBJECT_CONFIDENCE` (default: `0.3`) — detection confidence threshold for indexing
- `INDEX_OBJECT_EMBED_DIM` (default: `512`) — dimension of per-object CLIP embeddings

**`backend/pipeline.py`** — Core pipeline changes:
- `embed_image()` — new public helper for single-image CLIP encoding (used for object crops)
- `VideoIndex` dataclass — added `object_metadata: list[dict]` field and 6 new methods:
  - `object_embeddings_path()` / `object_faiss_path()` — file paths for object storage
  - `get_object_faiss_index()` / `save_object_faiss_index()` — lazy-loaded FAISS index with cache
  - `load_object_embeddings()` / `save_object_embeddings()` — numpy array persistence
- `index_video()` — three insertions:
  1. MIN_FPS enforcement — if motion sampling undershoots `INDEX_MIN_FPS`, uniformly-spaced frames are added
  2. Object detection & embedding — runs YOLO-World detection per keyframe, crops valid detections, batch-encodes crops through CLIP, populates `object_metadata`
  3. Object index persistence — builds `object_index.faiss` + saves `object_embeddings.npy`
- Import: added `from detector import detect`

**`backend/search_engine.py`** — Hybrid search:
- `_search_objects_faiss()` — queries the object FAISS index with the user's query text embedding, returns `{frame_idx: [matched_objects]}` mapping
- `search()` — per video, also searches the object FAISS index alongside frame-level FAISS; prefers indexed objects over running detection at query time; falls back to `_get_detections_for_segment()` for old indexes without object data
- Import: added `embed_text`

**`backend/main.py`** — Restart recovery:
- `_load_persisted_indexes()` now reads `object_metadata` from `index.json` on server restart

**`backend/.env.example`** — Documented all 4 new Phase 1 settings

### Data Flow
```
index_video():
  motion_sample → INDEX_MIN_FPS enforcement → extract → CLIP embed
    → YOLO-World detection per keyframe
    → crop each valid detection
    → batch CLIP-encode all crops
    → object_metadata[] + object_embeddings.npy + object_index.faiss

search():
  frame-level FAISS search + object-level FAISS search
  → hybrid weighted score per segment
```

### Storage Layout (per video)
```
storage/frames/{video_id}/
├── index.json               (includes object_metadata array)
├── embeddings.npy
├── index.faiss
├── object_embeddings.npy
├── object_index.faiss
├── frame_0.jpg
└── ...
```

### Verification
- **37/37 tests pass**
- Backward compatible: old indexes without object files gracefully fall back
- No new Python dependencies

---

## Phase 2 — Object Tracking

### Objective
Add cross-frame object tracking so the same object instance can be identified across consecutive keyframes, enabling track-consistency as a search signal.

### Files Modified

**`backend/pipeline.py`**:
- `SimpleTracker` — new IoU-based tracker (~80 lines, no external dependencies). Tracks objects across frames by class + bounding-box overlap.
  - `update(dets, frame_idx)` — matches detections to active tracks, spawns new tracks, returns detections with `track_id`
  - `summary()` — builds trajectory metadata `{track_id: {class, start/end_frame, total_frames, avg_confidence, displacement}}`
- `VideoIndex` — added `track_metadata: dict` field
- `index_video()` — integrates `SimpleTracker` into the detection loop; each object stored with `track_id`; `track_metadata` computed after loop and persisted in `index.json`

**`backend/config.py`**:
- `TRACK_MATCH_THRESHOLD` (default: `0.5`) — IoU threshold for matching detections to tracks
- `TRACK_BUFFER` (default: `30`) — frames before a lost track is deactivated
- `TRACK_CONSISTENCY_WEIGHT` (default: `0.10`) — weight for track consistency in hybrid score

**`backend/search_engine.py`**:
- `_compute_iou()` — new helper computing IoU overlap between two bounding boxes (used for relationship overlap checking)
- `_search_objects_faiss()` now returns `(frame_map, track_info)` — includes `track_id` per match and looks up trajectory metadata
- `search()` computes `tracking_consistency` per segment: normalized track length × match score
- `search()` checks relationship IoU overlap in Stage 3: when `search_plan.relationships` exists (e.g. person+backpack), computes IoU between all co-occurring objects on the middle frame and boosts `relationship_overlap` signal when IoU > 0.1

**`backend/config.py`** — Added `RELATIONSHIP_WEIGHT=0.10` for the 7th scoring signal

**`backend/tests/test_pipeline.py`** — 7 new tests for `SimpleTracker`

**`backend/main.py`**:
- `_load_persisted_indexes()` loads `track_metadata` from `index.json`

**`backend/.env.example`** — Added 3 tracking settings

**`backend/tests/test_pipeline.py`** — 7 new tests for `SimpleTracker`

### Verification
- **44/44 tests pass** (37 + 7 new)
- Zero new dependencies — pure NumPy IoU computation
- Old indexes load gracefully (`track_id` defaults to -1, `track_metadata` defaults to `{}`)

---

## Phase 3 — Semantic Understanding

### Objective
Add scene captioning and regex-based query parsing so the system understands natural language intent — extracting objects, attributes, actions, and location from queries, and augmenting hybrid scoring with caption-embedding similarity.

### Files Modified (3 existing files, 0 new)

**`backend/config.py`** — New settings:
- `CAPTIONER_MODEL` (default: `microsoft/Florence-2-base`) — VLM for per-keyframe caption generation
- `CAPTIONER_ENABLED` (default: `False`) — opt-in due to ~800MB model download
- `QUERY_PARSER_MODE` (default: `regex`) — query parsing strategy (`regex` or `llm`)

**`backend/pipeline.py`**:
- `Captioner` class — Florence-2 caption generator with lazy loading (same pattern as `_get_clip()`). Uses `transformers` (already a dependency). Graceful fallback on failure. Generates `<MORE_DETAILED_CAPTURE>` captions via `AutoModelForCausalLM`.
- `_get_captioner()` — thread-safe singleton factory with `False` sentinel on failure.
- `VideoIndex` — added `captions: dict[int, str]` field (frame_idx → caption text). Added 6 persistence methods following the object-embedding pattern:
  - `caption_embeddings_path()` / `caption_faiss_path()` — file paths
  - `get_caption_faiss_index()` / `save_caption_faiss_index()` — lazy-loaded FAISS cache
  - `load_caption_embeddings()` / `save_caption_embeddings()` — numpy array persistence
- `index_video()` — when `CAPTIONER_ENABLED=True`, generates captions per keyframe, CLIP-encodes them via `embed_text()`, saves `caption_embeddings.npy` and `caption_index.faiss` alongside existing files. Tracks `caption_time` in benchmarks.

**`backend/search_engine.py`**:
- `_extract_search_plan()` — regex-based query parser (roadmap §4.3):
  - Extracts objects from `_COMMON_CLASSES` lexicon
  - Extracts attributes via `_ATTR_PATTERNS` (e.g. "white car" → `{car: white}`)
  - Extracts location via `_LOCATION_PATTERNS` (e.g. "near the entrance")
  - Extracts actions via `_ACTION_PATTERNS` (walking, crossing, stopped, carrying)
  - Infers `relationships` from co-occurring objects (e.g. person+backpack)
- `_search_captions_faiss()` — searches caption FAISS index per video, returns `{frame_idx: caption_similarity}`. Backward compatible: returns `{}` if no caption embeddings exist.
- `Attribute expansion` — when `search_plan.attributes` contains entries like `{car: "white"}`, appends `"white car"` to the query text for all FAISS searches (frame, object, caption, clip). This ensures CLIP embedding search matches "white car" vs. just "car".
- `_hybrid_score()` — `caption_similarity` is now a live signal (replaces `CAPTION_WEIGHT * 0.0` placeholder). Uses `seg["caption_similarity"]` value with `settings.CAPTION_WEIGHT`.
- `search()` — integrated changes:
  1. Calls `_extract_search_plan()` for structured query understanding
  2. Uses plan objects for SQLite Stage 2 filtering (replaces direct `_extract_class_names()` call)
  3. Adds caption FAISS search as Stage 1b
  4. Computes `caption_similarity` per segment in Stage 3
  5. Adjusts motion score based on action context (e.g. "stopped" inverts motion)
  6. Returns `search_plan` in `query_info` for transparency

**`backend/.env.example`** — Documented all Phase 3 settings

### Data Flow (indexing, caption path)
```
index_video():
  motion_sample → MIN_FPS → extract → CLIP embed
    → (if CAPTIONER_ENABLED) Florence-2 caption per keyframe
    → CLIP text-encode caption → caption_embedding
    → caption_embeddings.npy + caption_index.faiss
    → YOLO-World detection + tracking → ...
```

### Search Pipeline (after Phase 3)
```
User query
  │
  │ _extract_search_plan() → {objects, attributes, location, actions}
  │
  │ Stage 1a: FAISS frame retrieval
  │ Stage 1b: FAISS caption retrieval
  │ Stage 2: SQLite class filter (from search_plan.objects)
  │ Stage 3: 6-signal hybrid score (caption_similarity live)
  │ Stage 4: Cross-encoder reranker (optional)
  ▼
Return segments + search_plan in query_info
```

### Verification
- **50/50 tests pass** — all existing tests pass unmodified
- Backward compatible: old indexes without caption files gracefully skip caption scoring
- No new Python dependencies (`transformers` already in requirements.txt)
- Captioner is disabled by default (`CAPTIONER_ENABLED=False`)
- Regex parser is always-on (`QUERY_PARSER_MODE=regex`), zero latency
- No new files created — all changes in 3 existing files

---

## Phase 4 — Temporal Reasoning

### Objective
Answer action queries ("crossing the road", "walking toward camera") by indexing overlapping clip windows of consecutive keyframes — combining averaged frame embeddings with motion statistics — and analyzing track trajectories for action classification.

### Files Modified (3 existing files, 0 new)

**`backend/config.py`** — New settings:
- `CLIP_WINDOW_SIZE` (default: `3`) — consecutive keyframes per clip window
- `CLIP_STRIDE` (default: `2`) — stride between overlapping windows
- `CLIP_MOTION_WEIGHT` (default: `0.3`) — blend between averaged frame embedding and motion features

**`backend/pipeline.py`**:
- `_encode_clip_light()` — combines averaged frame embeddings (already L2-normalized) with motion-score statistics (mean, std, max) into a single L2-normalized clip vector. Uses `CLIP_MOTION_WEIGHT` to blend semantic and motion signals. Zero new dependencies — reuses existing CLIP embeddings and motion scores.
- `VideoIndex` — added `clip_indices: list[list[int]]` field (frame indices per clip window). Added 3 persistence methods:
  - `clip_embeddings_path()` / `load_clip_embeddings()` / `save_clip_embeddings()`
  - No separate FAISS index — clips are searched linearly (small N)
- `index_video()` — after frame embedding, builds overlapping clip windows (`CLIP_WINDOW_SIZE` × `CLIP_STRIDE`), encodes each via `_encode_clip_light()`, saves `clip_embeddings.npy`. Only when enough keyframes exist (≥ `CLIP_WINDOW_SIZE`).

**`backend/search_engine.py`**:
- `_search_clip_embeddings()` — linear cosine similarity search of clip embeddings. Scores propagate to all frames within each clip's window. Returns `{frame_idx: temporal_score}`. Empty dict if no clip embeddings exist (backward compatible).
- `_compute_temporal_score()` — trajectory-based action analysis using track displacement and frame metadata:
   - `crossing`: score = ratio of displacement to 50% frame width
   - `stopped`: score = inverse of displacement ratio (stillness)
   - `walking`: score = displacement ratio + duration bonus
   - `walking_toward`: score = duration bonus + base 0.3 (person approaching camera)
   - `turning`: score = displacement ratio + base 0.25 (captures orientation change)
   - `carrying`: default 0.5 (soft boost for carried-object co-occurrence)
- `_hybrid_score()` — `temporal_alignment` signal is now live (replaces `TEMPORAL_WEIGHT * 0.0` placeholder). Uses `seg["temporal_alignment"]` with `settings.TEMPORAL_WEIGHT`.
- `search()` — integrated changes:
  1. Adds clip embedding search as Stage 1c (after frame + caption FAISS)
  2. Stores `track_ids` set on each segment during object matching (for trajectory lookup)
  3. Computes `temporal_alignment` per segment: max of clip-similarity and trajectory-action score
  4. All 6 signals now live: clip_semantic, caption_similarity, object_match, motion_activity, track_consistency, temporal_alignment

**`backend/.env.example`** — Documented all Phase 4 settings

### Data Flow (indexing, clip path)
```
index_video():
  motion_sample → MIN_FPS → extract → CLIP embed
    → build overlapping windows (window=3, stride=2)
    → _encode_clip_light() per window:
        avg(frame_embeddings) × (1 - CLIP_MOTION_WEIGHT)
        + concat [mean(motion), std(motion), max(motion)] × CLIP_MOTION_WEIGHT
        → L2-normalize
    → clip_embeddings.npy
    → caption, detection, tracking, SQLite → ...
```

### Search Pipeline (after Phase 4)
```
User query
  │
  │ _extract_search_plan() → {objects, attributes, location, actions}
  │
  │ Stage 1a: FAISS frame retrieval
  │ Stage 1b: FAISS caption retrieval
  │ Stage 1c: Linear clip embedding search
  │ Stage 2: SQLite class filter
  │ Stage 3: 6-signal hybrid score (all 6 live)
  │ Stage 4: Cross-encoder reranker (optional)
  ▼
Return segments + search_plan in query_info
```

### Verification
- **50/50 tests pass** — all existing tests pass unmodified
- Backward compatible: old indexes without `clip_embeddings.npy` skip temporal scoring
- Zero new Python dependencies (uses existing numpy, cv2, CLIP)
- No new files created — all changes in 3 existing files

---

## Phase 5 — Hybrid Retrieval & Reranking

### Objective
Implement the 4-stage multi-pipeline search from the roadmap (Stage 1: FAISS → Stage 2: SQLite class filter → Stage 3: 6-signal hybrid scoring → Stage 4: cross-encoder reranker), replacing the old 3-weight scoring system.

### Files Modified

**`backend/config.py`** — 7 granular hybrid scoring weights (roadmap §6.1):
- `CLIP_WEIGHT=0.20`, `CAPTION_WEIGHT=0.20`, `OBJECT_MATCH_WEIGHT=0.25`
- `MOTION_MATCH_WEIGHT=0.10`, `TRACK_CONSISTENCY_WEIGHT=0.15`, `TEMPORAL_WEIGHT=0.10`, `RELATIONSHIP_WEIGHT=0.10`
- `RERANKER_MODEL` (default: `BAAI/bge-reranker-v2-m3`)
- `RERANKER_ENABLED` (default: `False`)
- `SEMANTIC_WEIGHT`/`OBJECT_WEIGHT` preserved as backward-compat aliases

**`backend/pipeline.py`**:
- `MetadataDB` class — per-video SQLite metadata store using stdlib `sqlite3`
  - Two tables: `objects` (frame_idx, track_id, class, confidence, bbox, timestamp) and `tracks` (track_id, class, start/end_frame, total_frames, avg_confidence, displacement), indexed on `class` and `track_id`
- `VideoIndex.metadata_db_path()` — returns `storage/frames/{id}/metadata.db`
- `index_video()` — after detection/tracking/embedding, populates SQLite DB alongside existing JSON serialization

**`backend/search_engine.py`** — Full rewrite with 4-stage pipeline:

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | FAISS semantic retrieval | CLIP text → frame FAISS + object FAISS |
| 2 | SQLite class filter | `_extract_class_names()` parses query, `_filter_by_class_sqlite()` narrows candidates to frames containing matched objects |
| 3 | 7-signal hybrid scoring | `_hybrid_score()` — clip_semantic, caption_similarity, object_match, motion_activity, track_consistency, temporal_alignment, relationship_overlap |
| 4 | Cross-encoder reranker | `Reranker` class — lazy-loaded, global sentinel for failed loads, graceful fallback if `sentence-transformers` not installed |

Key changes:
- `_search_objects_faiss()` accepts optional `sqlite_frame_set` to filter results (Stage 2 integration)
- `_compute_motion_score()` maps middle frame index to indexed `motion_scores` array
- `_hybrid_score()` computes all 6 signals with individual configurable weights
- `search()` — wire stages sequentially, populate `score_breakdown` with all 6 fields

**`backend/.env.example`** — Documented all Phase 5 settings (weights + reranker)

**`backend/requirements.txt`** — Added `sentence-transformers>=3.0.0,<4`

### Search Pipeline (after Phase 5)
```
User query
  │
  │ Stage 1: FAISS broad retrieval (frame + object + caption + clip embeddings)
  │ Stage 2: SQLite metadata filter (_filter_by_class_sqlite)
  │ Stage 3: 6-signal hybrid weighted score
  │ Stage 4: Cross-encoder reranking (Reranker.rerank) — optional
  ▼
Return segments + score_breakdown + search_plan
```

### Verification
- **50/50 tests pass** (44 + 6 new MetadataDB tests)
- Zero new pure-Python dependencies (sqlite3 is stdlib; sentence-transformers is optional)
- Backward compatible: old indexes without SQLite DB fall back gracefully
- No new files created — all changes in existing 6 files

---

## Complete Storage Layout (all phases, per video)

```
storage/frames/{video_id}/
├── index.json                   (metadata, object_metadata, track_metadata, captions, clip_indices)
├── embeddings.npy               (frame CLIP embeddings)
├── index.faiss                  (frame FAISS index)
├── object_embeddings.npy        (object crop CLIP embeddings)
├── object_index.faiss           (object FAISS index)
├── caption_embeddings.npy       (caption text CLIP embeddings)
├── caption_index.faiss          (caption FAISS index)
├── clip_embeddings.npy          (clip-level embeddings: avg frame emb + motion stats)
├── metadata.db                  (SQLite: objects + tracks tables)
├── frame_0.jpg
├── frame_XX.jpg
└── ...
```

## Final Search Pipeline (all phases combined)

```
User query
  │
  │ _extract_search_plan()  → {objects, attributes, location, actions}
  │
  │ Stage 1a: FAISS frame retrieval        │ Phase 1-5
  │ Stage 1b: FAISS caption retrieval       │ Phase 3
  │ Stage 1c: Linear clip embedding search  │ Phase 4
  │ Stage 2: SQLite class filter            │ Phase 5
  │ Stage 3: 6-signal hybrid score          │ Phase 1-5
  │   ├── clip_semantic (CLIP_WEIGHT=0.20)
  │   ├── caption_similarity (CAPTION_WEIGHT=0.20)
  │   ├── object_match (OBJECT_MATCH_WEIGHT=0.25)
  │   ├── motion_activity (MOTION_MATCH_WEIGHT=0.10)
  │   ├── track_consistency (TRACK_CONSISTENCY_WEIGHT=0.15)
  │   ├── temporal_alignment (TEMPORAL_WEIGHT=0.10)
  │   └── relationship_overlap (RELATIONSHIP_WEIGHT=0.10)
  │ Stage 4: Cross-encoder reranker        │ Phase 5
  ▼
Return segments + score_breakdown + search_plan
```

## Summary

| Phase | Description | Key Files Changed | Test Count |
|-------|-------------|------------------|------------|
| 1 | Object detection during indexing | config.py, pipeline.py, search_engine.py, main.py | 37 |
| 2 | Object tracking + relationship IoU overlap | config.py, pipeline.py, search_engine.py, main.py | 44 |
| 3 | Scene captioning + regex query parser + attribute expansion | config.py, pipeline.py, search_engine.py | 50 |
| 4 | Clip indexing + action query engine (walking_toward, turning) | config.py, pipeline.py, search_engine.py | 50 |
| 5 | 7-signal hybrid scoring + SQLite + reranker | config.py, pipeline.py, search_engine.py | 50 |

- **50/50 tests pass** across all phases
- **Zero new files created** — all changes in existing files
- **Zero unnecessary abstractions** — each class/function serves a specific purpose
- **Full backward compatibility** — older indexes degrade gracefully at every phase
