# SceneTrace AI — Architecture & Implementation Summary

> Consolidated from `INDUSTRY_UPGRADE_ROADMAP.md`, `PHASE_IMPLEMENTATION_SUMMARY.md`, and `extra_impl.md`

## Storage Layout (per video)

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
└── frame_XX_d.jpg               (annotated thumbnails with green bounding boxes)
```

## Phase 1 — Object Detection During Indexing

**What changed:** YOLO-World detection moved from post-search into indexing. Each keyframe is detected, valid crops are CLIP-embedded, then stored in `object_metadata` + `object_index.faiss`.

**Files:** `config.py`, `pipeline.py`, `search_engine.py`, `main.py`
- `INDEX_MIN_FPS=5` — minimum 5fps after motion sampling; uniform frames inserted if undershot
- `INDEX_DETECTION_PROMPTS` — 80-class COCO prompt list
- `INDEX_OBJECT_CONFIDENCE=0.3` — detection threshold
- `INDEX_OBJECT_EMBED_DIM=512` — per-object CLIP embedding dimension
- `embed_image()` — single-image CLIP encoding
- `VideoIndex` — 6 new methods for object FAISS storage/persistence
- `_search_objects_faiss()` — queries object FAISS index at search time

## Phase 2 — Object Tracking & Relationship IoU

**What changed:** `SimpleTracker` (IoU-based, zero external dependencies) assigns persistent `track_id` across frames. Relationship IoU overlap checking added as a 7th scoring signal.

**Files:** `config.py`, `pipeline.py`, `search_engine.py`, `main.py`
- `SimpleTracker` — multi-class IoU tracking with configurable match threshold (0.5) and buffer (30 frames)
- `track_metadata` — start/end_frame, total_frames, displacement, avg_confidence
- `_compute_iou()` — IoU computation for relationship overlap
- Stage 3 checks `search_plan.relationships` (person+backpack, person+dog, person+bicycle) and computes IoU between co-occurring objects on the middle frame
- `RELATIONSHIP_WEIGHT=0.10` — 7th signal in hybrid scoring

## Phase 3 — Semantic Understanding

**What changed:** Florence-2 captioner generates per-keyframe captions (opt-in). Regex-based `_extract_search_plan()` parses natural language queries into structured plans. Attributes are expanded into the search query.

**Files:** `config.py`, `pipeline.py`, `search_engine.py`
- `Captioner` — Florence-2 with lazy loading, `MORE_DETAILED_CAPTURE` prompt
- `CAPTIONER_ENABLED=False` (opt-in, ~800MB model)
- `caption_embeddings.npy` + `caption_index.faiss`
- `_extract_search_plan()` — regex parser extracting: objects (80-class lexicon), attributes (color/type), location, actions (6 types), relationships
- **Attribute expansion:** `search_plan.attributes` (e.g. `{car:"white"}`) appended to query text for all FAISS searches
- `CAPTION_WEIGHT=0.20` — caption similarity live in hybrid score

## Phase 4 — Temporal Reasoning

**What changed:** Overlapping clip windows combine averaged frame embeddings + motion statistics. Track trajectories are analyzed for action classification. `walking_toward` and `turning` actions added.

**Files:** `config.py`, `pipeline.py`, `search_engine.py`
- `_encode_clip_light()` — blends avg frame embeddings with motion stats (mean, std, max), L2-normalized
- `CLIP_WINDOW_SIZE=3`, `CLIP_STRIDE=2`, `CLIP_MOTION_WEIGHT=0.3`
- `clip_embeddings.npy` — linear cosine similarity search
- `_compute_temporal_score()` — 6 actions:
  - `crossing` — displacement > 50% frame width
  - `stopped` — inverse displacement (stillness)
  - `walking` — displacement + duration bonus
  - `walking_toward` — duration bonus + 0.3 base
  - `turning` — displacement + 0.25 base
  - `carrying` — default 0.5 soft boost
- All 6 signals live: `TEMPORAL_WEIGHT=0.10`

## Phase 5 — Hybrid Retrieval & Reranking

**What changed:** 4-stage pipeline: Stage 1 (FAISS) → Stage 2 (SQLite) → Stage 3 (7-signal hybrid scoring) → Stage 4 (cross-encoder reranker). Replaced old 3-weight scoring.

**Files:** `config.py`, `pipeline.py`, `search_engine.py`
- `MetadataDB` (SQLite, stdlib) — objects + tracks tables, indexed on class and track_id
- `Reranker` — `BAAI/bge-reranker-v2-m3`, lazy-loaded, `False` sentinel for graceful fallback
- `_hybrid_score()` — 7 signals:
  1. `clip_semantic` (0.20)
  2. `caption_similarity` (0.20)
  3. `object_match` (0.25)
  4. `motion_activity` (0.10)
  5. `tracking_consistency` (0.15)
  6. `temporal_alignment` (0.10)
  7. `relationship_overlap` (0.10)

## Visualization — Annotated Thumbnails

**What changed:** Search segments now include `annotated_thumbnail` URL pointing to the middle frame rendered with green bounding boxes, class labels, and confidence scores.

**Files:** `detector.py`, `search_engine.py`
- `render()` — draws green bboxes with labels using OpenCV
- `_save_annotated_thumbnail()` — saves `frame_{mid}_d.jpg` for both pre-indexed objects and query-time detection
- Consecutive frames merged into segments via `frames_to_segments()`
- Video clip extraction endpoint `/api/clips/{video_id}` generates MP4 clips

## Final Search Pipeline

```
User query
  │
  │ _extract_search_plan() → {objects, attributes, location, actions, relationships}
  │
  │ Stage 1a: FAISS frame retrieval
  │ Stage 1b: FAISS caption retrieval
  │ Stage 1c: Linear clip embedding search
  │ Stage 2: SQLite class filter
  │ Stage 3: 7-signal hybrid score
  │   ├── clip_semantic (CLIP_WEIGHT=0.20)
  │   ├── caption_similarity (CAPTION_WEIGHT=0.20)
  │   ├── object_match (OBJECT_MATCH_WEIGHT=0.25)
  │   ├── motion_activity (MOTION_MATCH_WEIGHT=0.10)
  │   ├── track_consistency (TRACK_CONSISTENCY_WEIGHT=0.15)
  │   ├── temporal_alignment (TEMPORAL_WEIGHT=0.10)
  │   └── relationship_overlap (RELATIONSHIP_WEIGHT=0.10)
  │ Stage 4: Cross-encoder reranker (optional)
  ▼
Return segments + score_breakdown + search_plan + annotated_thumbnails
```

## Gap Analysis (extra_impl.md items)

| # | Item | Status |
|---|------|--------|
| 1 | Object-Level Indexing | ✅ Implemented (Phase 1) |
| 2 | Adaptive Frame Sampling | ✅ Implemented (INDEX_MIN_FPS=5) |
| 3 | Persistent Tracking | ✅ Implemented (SimpleTracker) |
| 4 | Detection During Indexing | ✅ Implemented (YOLO-World) |
| 5 | Rich Metadata | ✅ Implemented (object_metadata + SQLite) |
| 6 | LLM Query Parser | ⚠️ Partially (regex parser exists, LLM path unimplemented) |
| 7 | Scene Captions | ✅ Implemented (Florence-2, opt-in) |
| 8 | Hybrid Retrieval | ✅ Implemented (7 signals) |
| 9 | Cross-Encoder Reranking | ✅ Implemented (BAAI/bge-reranker-v2-m3) |
| 10 | Temporal Reasoning | ✅ Implemented (clip embeddings + trajectory analysis) |
| 11 | Multi-Object/Relationship Queries | ✅ Implemented (IoU overlap + RELATIONSHIP_WEIGHT) |
| 12 | Visualization | ✅ Implemented (annotated thumbnails in search results) |
| 13 | Pipeline Architecture | ⚠️ Mostly matched (YOLO-World/SimpleTracker instead of YOLOv11/ByteTrack) |
| 14 | Model Recommendations | 📋 Recommendations only |
| 15 | Expected Outcome | ✅ Achieved |

## Verification

- **50/50 tests pass** across all phases
- **Zero new files created** during phases — all changes in existing files
- **Full backward compatibility** — older indexes degrade gracefully at every phase
- **Zero unnecessary abstractions** — each class/function serves a specific purpose
