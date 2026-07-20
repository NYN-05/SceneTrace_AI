# Sequential Processing Bottleneck Analysis

## Overview

The codebase has three main execution paths: (A) video indexing in `pipeline.py`, (B) search in `search_engine.py`, and (C) API orchestration in `main.py`. Below is every sequential bottleneck identified, prioritized by impact.

---

## PRIORITY 0 — Critical (largest speedup, no accuracy impact)

### 1. Captioning loop — per-frame sequential Florence-2 inference

- **File:** `pipeline.py`, lines 688-696
- **What it does:**
  ```python
  for fi_idx, (frame_idx, frame) in enumerate(zip(keep_frames, frames)):
      cap = captioner.caption(frame)   # Florence-2 autoregressive generation
      captions[frame_idx] = cap
      caption_texts.append(cap)
  ```
  Each frame is captioned one-at-a-time through Florence-2 (a 0.2B-0.8B parameter causal LM). The `caption()` method (lines 275-291) runs `model.generate()` with `num_beams=3` and `max_new_tokens=512`, which is the single most expensive per-frame operation in the entire pipeline.
- **Can it be parallelized?** Yes. Each frame's caption is independent. Use a `ThreadPoolExecutor` to submit N frames concurrently. Florence-2 on GPU will serialize internally (CUDA kernel launches are sequential per device), but the CPU-side tokenization, beam-search scheduling, and KV-cache management can overlap. For CPU-only inference, this is a pure Nx speedup.
- **Accuracy impact:** None -- each frame is captioned identically regardless of ordering.
- **Speedup estimate:** 2-4x with 4 workers on GPU (CPU preprocessing + GPU kernel launch overlap); up to Nx on CPU-only.

### 2. Object detection per-frame loop

- **File:** `pipeline.py`, lines 714-734
- **What it does:**
  ```python
  for fi_idx, (frame_idx, frame) in enumerate(zip(keep_frames, frames)):
      dets = detect(frame, settings.INDEX_DETECTION_PROMPTS, ...)  # YOLO-World inference
      tracked = tracker.update(dets, frame_idx)  # stateful tracker
  ```
  The `detect()` call (line 716) runs YOLO-World inference on each frame sequentially. The `tracker.update()` call (line 719) has state dependency (it maintains `self._tracks` across frames).
- **Can it be parallelized?** Partially. The `detect()` calls are independent per frame and can run in parallel via `ThreadPoolExecutor`. The `tracker.update()` must remain sequential because `SimpleTracker` maintains state (`self._tracks`, `self._next_id`, `self._frame_num`). The pattern would be: parallelize all detections first, then run tracking sequentially over the results.
- **Accuracy impact:** None -- detection results are identical regardless of order. Tracking is then applied to the same detections.
- **Speedup estimate:** 3-5x for the detection portion (which dominates the loop time). The tracking step is cheap (IoU computation).

### 3. Three independent FAISS index builds

- **File:** `pipeline.py`, lines 765-793
- **What it does sequentially:**
  ```python
  # Line 765: Main FAISS index
  faiss_index = build_faiss_index(embs)

  # Lines 777-781: Object FAISS index
  object_faiss_index = build_faiss_index(object_embs)
  idx.save_object_faiss_index(object_faiss_index)
  idx.save_object_embeddings(object_embs)

  # Lines 783-788: Caption FAISS index
  caption_faiss_index = build_faiss_index(caption_embs)
  idx.save_caption_faiss_index(caption_faiss_index)
  idx.save_caption_embeddings(caption_embs)
  ```
  Three `build_faiss_index()` calls run sequentially. Each is CPU-bound (k-means clustering for IVF, vector adds). They operate on completely independent data (`embs`, `object_embs`, `caption_embs`).
- **Can it be parallelized?** Yes. Use `ThreadPoolExecutor` to run all three builds in parallel. The FAISS operations release the GIL (they are C++ extensions), so threads are effective.
- **Accuracy impact:** None -- each index is built on independent data.
- **Speedup estimate:** Up to 3x for this section (wall-clock time drops from t_main + t_obj + t_cap to max(t_main, t_obj, t_cap)).

### 4. Search engine — three independent per-video loops

- **File:** `search_engine.py`, lines 454-496
- **What it does sequentially:**
  ```python
  # Loop 1 (lines 454-482): FAISS semantic retrieval + object search per video
  for vid, idx in indexes.items():
      ...search_embeddings()...
      ..._search_objects_faiss()...

  # Loop 2 (lines 486-489): Caption FAISS search per video
  for vid, idx in indexes.items():
      cap_sims = _search_captions_faiss(expanded_query, idx)

  # Loop 3 (lines 492-496): Clip embedding search per video
  for vid, idx in indexes.items():
      clip_sims = _search_clip_embeddings(expanded_query, idx)
  ```
  Three sequential loops over all indexed videos. Each iteration within a loop is independent per video. Moreover, the three loops are independent of each other (no data dependency between semantic search, caption search, and clip search).
- **Can it be parallelized?** Yes, in two dimensions:
  1. **Per-video parallelism:** Use `ThreadPoolExecutor` to process multiple videos concurrently within each loop.
  2. **Cross-loop parallelism:** All three loops can run concurrently since they produce independent results (`candidates`, `caption_frame_sims`, `clip_frame_sims`) that are only merged later in Stage 3 (line 499+).
- **Accuracy impact:** None -- all operations are read-only queries against independent indexes.
- **Speedup estimate:** For N videos, up to Nx per loop, and 3x from running loops in parallel. Realistically 3-6x for typical multi-video deployments.

---

## PRIORITY 1 — High (significant speedup, easy to implement)

### 5. Per-segment scoring loop

- **File:** `search_engine.py`, lines 499-594
- **What it does:**
  ```python
  for seg in candidates:   # sequential over all candidate segments
      # compute object_score, tracking_consistency, relationship_overlap,
      # caption_similarity, temporal_alignment, motion_activity
      # then _hybrid_score()
  ```
  Each iteration reads from pre-computed dictionaries (`object_frame_maps`, `caption_frame_sims`, `clip_frame_sims`, `object_track_info`) and writes to the segment dict. There are no cross-segment dependencies.
- **Can it be parallelized?** Yes. Use `ThreadPoolExecutor` to process segments in parallel. Each segment's scoring is a pure read from shared dicts + write to its own dict.
- **Accuracy impact:** None -- the hybrid score computation is deterministic and order-independent.
- **Speedup estimate:** 2-4x for typical top_k values (10-20 segments).

### 6. Per-video search loop in legacy API

- **File:** `main.py`, lines 234-249
- **What it does:**
  ```python
  for video_id, idx in targets.items():
      embs = idx.load_embeddings()
      faiss_idx = idx.get_faiss_index()
      indices, scores = await loop.run_in_executor(executor, search_embeddings, ...)
      segs = frames_to_segments(indices, scores)
      ...
  ```
  Each video's search is independent. Currently processed sequentially.
- **Can it be parallelized?** Yes. Use `asyncio.gather()` with `run_in_executor()` for each video. FAISS operations release the GIL.
- **Accuracy impact:** None.
- **Speedup estimate:** Nx for N videos (e.g., 3x for 3 videos).

### 7. CPU preprocessing in compute_embeddings

- **File:** `pipeline.py`, lines 199-200
- **What it does:**
  ```python
  rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]
  inputs = processor(images=rgb, return_tensors="pt", padding=True).to(device)
  ```
  The BGR→RGB conversion and CLIP processor (resize, normalize, tensor conversion) are CPU-bound and run sequentially per batch. The GPU is idle during this time.
- **Can it be parallelized?** Yes. Use `ThreadPoolExecutor` to preprocess the next batch while the current batch is on GPU. This is a producer-consumer pattern (double-buffering).
- **Accuracy impact:** None.
- **Speedup estimate:** 1.5-2x for the embedding phase by overlapping CPU preprocessing with GPU inference.

### 8. MetadataDB.populate() — sequential INSERTs

- **File:** `pipeline.py`, lines 422-441
- **What it does:**
  ```python
  for obj in object_metadata:          # line 425
      self.conn.execute("INSERT INTO objects ...", ...)
  for tid_str, td in track_metadata.items():  # line 434
      self.conn.execute("INSERT OR REPLACE INTO tracks ...", ...)
  ```
  Each object gets an individual `INSERT`. For thousands of objects, this is many round-trips to SQLite.
- **Can it be parallelized?** Not with threads (SQLite connections are not thread-safe by default), but can use `executemany()` to batch all object INSERTs into a single operation and all track INSERTs into another.
- **Accuracy impact:** None.
- **Speedup estimate:** 5-10x for the populate step (reduces N round-trips to 2).

---

## PRIORITY 2 — Medium (moderate speedup, low effort)

### 9. main.py — _load_persisted_indexes()

- **File:** `main.py`, lines 47-70
- **What it does:**
  ```python
  for idx_dir in frames_dir.iterdir():
      idx_file = idx_dir / "index.json"
      if idx_file.exists():
          data = json.loads(idx_file.read_text())
          vi = VideoIndex(...)
          _indexes[vi.video_id] = vi
  ```
  Each index is loaded sequentially from disk. Each load is independent.
- **Can it be parallelized?** Yes. Use `ThreadPoolExecutor` to load multiple indexes concurrently. JSON parsing is CPU-bound but releases the GIL during file I/O.
- **Accuracy impact:** None.
- **Speedup estimate:** 2-4x for directories with many indexes.

### 10. compute_embeddings() batch size

- **File:** `pipeline.py`, lines 191-212
- **Current default:** `BATCH_SIZE=32` (config.py:41)
- For CLIP-ViT-B/32 on a modern GPU, batch sizes of 64-128 would improve GPU utilization without OOM.
- Larger batches mean fewer kernel launches and better tensor core utilization.
- **No accuracy impact.**

### 11. Clip embedding loop

- **File:** `pipeline.py`, lines 665-671
- **What it does:**
  ```python
  for i in range(0, max(1, len(keep_frames) - w + 1), stride):
      win_frame_idxs = keep_frames[i:i + w]
      win_embs = embs[i:i + w]
      win_motion = motion_scores[i:i + w]
      clip_indices.append(win_frame_idxs)
      clip_embs_list.append(_encode_clip_light(win_embs, win_motion))
  ```
  Each iteration is independent. `_encode_clip_light` is pure numpy (mean, std, max, concat, norm). This is lightweight but for thousands of frames with small window/stride, it can add up.
- **Can it be parallelized?** Yes, with `ThreadPoolExecutor`. Numpy operations release the GIL.
- **Accuracy impact:** None.
- **Speedup estimate:** 2-3x for this section, but this section is already fast relative to model inference.

### 12. main.py — get_timeline()

- **File:** `main.py`, lines 331-338
- **What it does:**
  ```python
  for i, (fi, ts, ms) in enumerate(zip(...)):
      events.append({...})
  ```
  Lightweight loop, **not worth parallelizing.**

### 13. main.py — get_video_objects()

- **File:** `main.py`, lines 354-361
- **What it does:**
  ```python
  for fi in idx.frame_indices[:50]:
      annotated_path = vf / f"frame_{fi}_d.jpg"
      if annotated_path.exists():
          ...
  ```
  Each file existence check is independent. Could use `ThreadPoolExecutor` but the speedup is negligible (50 file stat calls).

---

## PRIORITY 3 — Low (small speedup, or requires careful design)

### 14. pipeline.py — motion_sample()

- **File:** `pipeline.py`, lines 473-526
- **What it does:** Sequential frame-by-frame reading from video, computing optical flow or frame difference. Has temporal dependency (`prev_gray` needed for each step).
- **Can it be parallelized?** No -- each frame depends on the previous frame's grayscale image. Optical flow is inherently sequential. The video decoder (`cap.read()`) is also sequential.
- **Accuracy impact:** Would change if parallelized (optical flow would be incorrect).

### 15. search_engine.py — _search_objects_faiss()

- **File:** `search_engine.py`, lines 244-295
- Already covered by #3 (per-video loop parallelization).

### 16. pipeline.py — frames_to_segments()

- **File:** `pipeline.py`, lines 576-590
- Lightweight; **not worth parallelizing.**

---

## Lock Contention Analysis

### `_indexes_lock`
- **File:** `main.py:33`, used at lines 67, 110, 163, 181, 201, 221, 229, 254, 267, 301, 308, 327, 348, 367
- Used in both search and index paths. During indexing (which runs in background thread), the lock is held briefly to add the result. During search (API handler), it's held to read the indexes dict. Contention is low because indexing is rare relative to searches.

### `_index_progress_lock`
- **File:** `main.py:34`, used at lines 154, 160, 165, 169, 178, 198
- Held during progress updates from the indexing thread and reads from the SSE stream. The SSE stream polls every 0.5s. Contention is low.

### `_faiss_cache_lock`
- **File:** `pipeline.py:25`, used at lines 49, 55, 64, 69, 91, 97, 106, 129, 135, 143
- Held during FAISS index cache reads/writes. Contention could increase if multiple searches run concurrently. The lock is held briefly (dict lookup/insert), so contention is low.

### `_clip_lock`
- **File:** `pipeline.py:177`, used at line 182
- Held only during CLIP model lazy-loading. Once loaded, never contended.

### `_CAPTIONER_LOCK`
- **File:** `pipeline.py:254`, used at line 297
- Held only during captioner lazy-loading. Once loaded, never contended.

### `_yolo_manager_lock`
- **File:** `detector.py:20`, used at line 103
- Held only during YOLO manager lazy-loading. Once loaded, never contended.

### `_RERANKER_LOCK`
- **File:** `search_engine.py:22`, used at line 38
- Held only during reranker lazy-loading. Once loaded, never contended.

### Conclusion on locks
No significant contention issues. All locks are used for lazy initialization or brief dict operations.

---

## Batch Size Analysis

### `compute_embeddings()` (pipeline.py:191-212)
- Current default `BATCH_SIZE=32` (config.py:41)
- For CLIP-ViT-B/32 on a modern GPU, batch sizes of 64-128 would improve GPU utilization without OOM
- Larger batches mean fewer kernel launches and better tensor core utilization
- No accuracy impact

### `embed_text()` (pipeline.py:215-220)
- Always called with a single text query. For caption embedding (line 699), all captions are embedded in one call. This is already batched.

---

## Prioritized Optimization Opportunities

### P0 — Critical (implement immediately)

| # | Location | Lines | Description | Parallelization Strategy | Speedup |
|---|----------|-------|-------------|--------------------------|---------|
| 1 | `pipeline.py` | 688-696 | Per-frame captioning loop — Florence-2 `model.generate()` called sequentially for each keyframe | `ThreadPoolExecutor` with 2-4 workers. Each frame's caption is independent. GPU serializes inference but CPU preprocessing and beam-search scheduling overlap. | 3-5x for this phase (often the slowest phase) |
| 2 | `pipeline.py` | 714-734 | Per-frame object detection loop — YOLO-World `detect()` called sequentially for each keyframe | Split into two phases: (a) parallelize all `detect()` calls with `ThreadPoolExecutor`, then (b) run `tracker.update()` sequentially over the results. Detection is ~95% of the loop time. | 3-5x for detection portion |
| 3 | `pipeline.py` | 765-793 | Three independent FAISS index builds — main embeddings, object embeddings, caption embeddings built sequentially | `ThreadPoolExecutor` with 3 workers. FAISS C++ code releases the GIL. | 2-3x for this section (wall time = max of 3, not sum) |
| 4 | `search_engine.py` | 454-496 | Three sequential per-video loops — semantic search, caption search, clip search over all videos | Two levels: (a) run all three loops in parallel via `ThreadPoolExecutor`, (b) within each loop, process videos in parallel. | 3-6x for multi-video searches |

### P1 — High (significant speedup, easy to implement)

| # | Location | Lines | Description | Parallelization Strategy | Speedup |
|---|----------|-------|-------------|--------------------------|---------|
| 5 | `search_engine.py` | 499-594 | Per-segment scoring loop — 6-signal hybrid score computed sequentially for each candidate segment | `ThreadPoolExecutor` to process segments in parallel. Each segment reads from pre-computed dicts and writes to its own dict. | 2-4x |
| 6 | `main.py` | 234-249 | Per-video search loop in legacy API — each video's FAISS search runs sequentially | `asyncio.gather()` with `run_in_executor()` per video. FAISS releases GIL. | Nx for N videos |
| 7 | `pipeline.py` | 199-200 | CPU preprocessing in embedding loop — BGR→RGB conversion + CLIP processor runs sequentially per batch while GPU is idle | Double-buffering: preprocess next batch in a thread while current batch is on GPU. Use `ThreadPoolExecutor` with a queue of size 2. | 1.5-2x |
| 8 | `pipeline.py` | 422-441 | `MetadataDB.populate()` — individual INSERTs for each object and track | Replace with `executemany()` for batch INSERTs. SQLite WAL mode already enabled. | 5-10x |

### P2 — Medium (moderate speedup, low effort)

| # | Location | Lines | Description | Parallelization Strategy | Speedup |
|---|----------|-------|-------------|--------------------------|---------|
| 9 | `main.py` | 47-70 | `_load_persisted_indexes()` — sequential JSON loads from disk | `ThreadPoolExecutor` to load indexes in parallel. JSON parsing is CPU-bound. | 2-4x at startup |
| 10 | `pipeline.py` | 191-212 | `compute_embeddings()` batch size — default BATCH_SIZE=32 may underutilize GPU | Increase BATCH_SIZE to 64-128 (depends on GPU VRAM). Fewer kernel launches, better tensor core utilization. | 1.3-2x for embedding phase |
| 11 | `pipeline.py` | 665-671 | Clip embedding loop — lightweight numpy per sliding window | `ThreadPoolExecutor` for `_encode_clip_light()` calls. Numpy releases GIL. | 2-3x for this section (already fast) |
| 12 | `main.py` | 331-338 | `get_timeline()` — sequential event list construction | Trivial; not worth parallelizing. | Negligible |
| 13 | `main.py` | 354-361 | `get_video_objects()` — sequential file existence checks | `ThreadPoolExecutor` for `Path.exists()` calls. | 1.5-2x (but already fast) |

### P3 — Low (small speedup, or requires careful design)

| # | Location | Lines | Description | Parallelization Strategy |
|---|----------|-------|-------------|--------------------------|
| 14 | `pipeline.py` | 473-526 | `motion_sample()` — sequential frame-by-frame optical flow | Cannot parallelize without changing algorithm (temporal dependency on `prev_gray`). Could use a different algorithm (e.g., frame-difference on GPU via CuPy). |
| 15 | `search_engine.py` | 244-295 | `_search_objects_faiss()` — per-video object FAISS search | Already covered by #3 (per-video loop parallelization). |
| 16 | `pipeline.py` | 576-590 | `frames_to_segments()` — sequential segment building | Lightweight; not worth parallelizing. |

---

## Summary of Maximum Theoretical Speedup

If all P0 and P1 optimizations were applied to a typical indexing pipeline (e.g., 1000 keyframes, 3 videos in search):

| Phase | Current (sequential) | Optimized (parallel) |
|-------|---------------------|---------------------|
| Captioning (1000 frames) | ~500s (0.5s/frame) | ~125s (4 workers) |
| Object detection (1000 frames) | ~200s (0.2s/frame) | ~50s (4 workers) |
| FAISS index builds (3 indexes) | ~6s (2+2+2) | ~2s (max of 3) |
| Search (3 videos, 3 signal types) | ~3s (9 sequential ops) | ~0.5s (parallel) |
| Embedding preprocessing | ~30s | ~15s (double-buffering) |
| MetadataDB populate | ~2s | ~0.3s (executemany) |
| **Total indexing pipeline** | **~732s** | **~192s** |
| **Total search (3 videos)** | **~3s** | **~0.5s** |

The single biggest win is parallelizing the captioning and object detection loops (P0 #1 and #2), which together account for ~95% of indexing time. The search engine per-video loop parallelization (P0 #4) is the biggest win for query latency.
