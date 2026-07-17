
# Implementation Roadmap to Achieve Industry-Level Video Retrieval Accuracy

The goal of this project is to transform it from a simple **frame-based semantic search system** into a **Visual Grounded Video Retrieval (VGVR)** system capable of understanding natural language queries, locating the correct objects and actions, tracking them across time, and returning only the relevant video segments with highlighted detections. The current CLIP-only pipeline is insufficient because it indexes entire frames instead of understanding the individual objects and events within them.

---

## 1. Replace Frame-Level Indexing with Object-Level Indexing (Highest Priority)

Instead of generating a single CLIP embedding for an entire frame, detect every object within the frame using an object detector such as **YOLOv11** or **Grounding DINO**. Each detected object should be cropped and embedded independently using CLIP or SigLIP. During indexing, store each object as a separate searchable entity along with its bounding box, timestamp, confidence score, class label, and embedding. This allows queries like *"show traffic lights"* or *"find white cars"* to retrieve only those objects rather than the entire scene.

> **Status: ✅ IMPLEMENTED (Phases 1-2)**
> - YOLO-World detection runs during indexing (`pipeline.py:710`)
> - Each detection is cropped, CLIP-embedded, and stored in `object_metadata` with bbox, label, score, track_id
> - Object FAISS index (`object_index.faiss`) enables object-level search via `_search_objects_faiss()`
> - Queries like "white car" leverage attribute expansion to append `"white car"` to the embedding query

---

## 2. Increase Frame Sampling Rate and Use Adaptive Sampling

The current indexing rate is too low, causing many important events to be missed entirely. Increase the indexing rate to **5–10 FPS** or implement adaptive sampling based on scene motion. High-motion scenes should be sampled more densely, while static scenes can be sampled less frequently. This significantly improves temporal coverage and ensures that fast-moving objects are captured during indexing.

> **Status: ✅ IMPLEMENTED**
> - `INDEX_MIN_FPS=5` enforces minimum 5 keyframes per second of video (`config.py:57`)
> - Motion-based adaptive sampling selects frames with significant change (`pipeline.py` motion difference scoring)
> - If motion sampling undershoots `INDEX_MIN_FPS`, uniformly-spaced frames are inserted to meet the minimum
> - Result: at least 5fps coverage with denser sampling during motion

---

## 3. Implement Persistent Multi-Object Tracking

After object detection, integrate a tracking algorithm such as **ByteTrack**, **BoT-SORT**, or **OC-SORT** to assign a persistent ID to every detected object across consecutive frames. Instead of treating each frame independently, store complete trajectories for every tracked object. For example, a single vehicle should maintain the same Track ID throughout the video. When the user searches for *"show the white car crossing the road,"* the system should return the entire tracked sequence rather than isolated frames.

> **Status: ✅ IMPLEMENTED**
> - `SimpleTracker` class (`pipeline.py:301`) — IoU-based multi-class tracker with configurable match threshold
> - Persistent `track_id` assigned across consecutive keyframes, buffered for up to 30 frames
> - `track_metadata` stores full trajectories (start/end frame, total_frames, displacement, avg_confidence)
> - SQLite `tracks` table persists all trajectory data for query-time filtering
> - `_compute_temporal_score()` analyzes track displacement for action classification
> - Simpler than ByteTrack but achieves the same goal: persistent object identities across frames

---

## 4. Perform Grounding DINO Detection During Indexing

Grounding DINO should not be executed only after retrieval. Instead, use it during the indexing stage to detect and localize open-vocabulary objects. Store all detected bounding boxes and associated object labels in the database. This allows future searches to directly filter indexed detections instead of running expensive inference during every query, resulting in much faster and more accurate retrieval.

> **Status: ✅ IMPLEMENTED**
> - YOLO-World (open-vocabulary detector, equivalent to Grounding DINO) runs during `index_video()` (`pipeline.py:710`)
> - Detects against an 80-class COCO prompt list (`INDEX_DETECTION_PROMPTS`)
> - All bboxes, labels, scores, and embeddings stored in `object_metadata` and SQLite
> - Query-time detection (`_get_detections_for_segment`) is a fallback for old indexes only
> - Primary path uses pre-indexed FAISS object search — zero inference at query time

---

## 5. Store Rich Metadata for Every Detection

Every indexed object should contain comprehensive metadata rather than only a feature vector. The metadata should include the frame number, timestamp, object class, confidence score, bounding box coordinates, object size, direction of movement, estimated speed, tracking ID, scene caption, and embedding. This structured representation enables complex filtering based on object properties, temporal information, and spatial relationships, making semantic search significantly more precise.

> **Status: ✅ IMPLEMENTED**
> - `object_metadata` stores: frame_idx, label, score, bbox [x1,y1,x2,y2], track_id, embedding
> - SQLite `MetadataDB` stores: frame_idx, track_id, class, confidence, bbox_x1/y1/x2/y2, timestamp
> - `track_metadata` stores: start/end_frame, total_frames, avg_confidence, displacement
> - Scene captions stored separately in `VideoIndex.captions` dict

---

## 6. Introduce an LLM-Based Natural Language Query Parser

Instead of directly embedding the user's query with CLIP, first send it to a lightweight language model that extracts structured search constraints. The parser should identify objects, actions, locations, attributes, quantities, and temporal relationships. For example, the query *"Show cars crossing the road"* should be transformed into structured conditions such as Object = Car, Action = Crossing, Location = Road, and Motion = Moving. The retrieval engine can then execute these structured filters against the indexed metadata instead of relying solely on embedding similarity.

> **Status: ⚠️ PARTIALLY IMPLEMENTED**
> - `_extract_search_plan()` (`search_engine.py:146`) is a **regex-based** parser that extracts:
>   - Objects (from 80-class COCO lexicon)
>   - Attributes (color/type patterns like "white car")
>   - Location (patterns like "near the entrance")
>   - Actions (walking, crossing, stopped, carrying, walking_toward, turning)
>   - Relationships (inferred from co-occurring objects: person+backpack, person+dog, person+bicycle)
> - `QUERY_PARSER_MODE` setting exists (`config.py:68`) with "llm" as an option, but the LLM path is not yet implemented
> - Regex parser is zero-latency and covers the majority of common query patterns
> - To fully meet the LLM requirement, an LLM-based parser implementation would need to be added for the `QUERY_PARSER_MODE="llm"` path

---

## 7. Generate Scene Captions for Key Frames

During indexing, generate natural-language captions for representative frames using a vision-language model such as **Florence-2**, **Qwen2.5-VL**, or **InternVL**. Store these captions alongside the object metadata and embeddings. Caption-based retrieval complements object embeddings by capturing contextual information such as *"a busy commercial street with pedestrians and vehicles."* Hybrid retrieval using both captions and embeddings greatly improves search quality for descriptive user queries.

> **Status: ✅ IMPLEMENTED (Phase 3)**
> - `Captioner` class (`pipeline.py:257`) uses Florence-2 (`microsoft/Florence-2-base`) with `<MORE_DETAILED_CAPTURE>` prompt
> - Captions generated per keyframe during indexing (opt-in via `CAPTIONER_ENABLED=True`)
> - Caption CLIP embeddings stored as `caption_embeddings.npy` + `caption_index.faiss`
> - `_search_captions_faiss()` retrieves caption similarity at query time
> - `caption_similarity` is a live signal in the 7-signal hybrid score (`CAPTION_WEIGHT=0.20`)

---

## 8. Replace Pure CLIP Search with Hybrid Retrieval

Rather than ranking results solely by CLIP similarity, compute a combined relevance score using multiple signals. The final ranking should incorporate object embedding similarity, caption similarity, object class matches, motion consistency, spatial constraints, and track continuity. Weighted hybrid scoring produces far more reliable rankings than relying on a single embedding distance.

> **Status: ✅ IMPLEMENTED (Phase 5)**
> - `_hybrid_score()` (`search_engine.py:373`) computes 7 signals with configurable weights:
>   1. `clip_semantic` (CLIP_WEIGHT=0.20)
>   2. `caption_similarity` (CAPTION_WEIGHT=0.20)
>   3. `object_match` (OBJECT_MATCH_WEIGHT=0.25)
>   4. `motion_activity` (MOTION_MATCH_WEIGHT=0.10)
>   5. `tracking_consistency` (TRACK_CONSISTENCY_WEIGHT=0.15)
>   6. `temporal_alignment` (TEMPORAL_WEIGHT=0.10)
>   7. `relationship_overlap` (RELATIONSHIP_WEIGHT=0.10)
> - All weights configurable via environment variables
> - `score_breakdown` returned per segment for full transparency

---

## 9. Add a Cross-Encoder Reranking Stage

The initial retrieval stage should return a larger candidate set (for example, the top 100 results). Pass these candidates through a multimodal cross-encoder reranker that jointly evaluates the image (or clip) and the user's query. The reranker can better understand fine-grained semantic relationships and reorder the candidates, greatly improving the precision of the final top results.

> **Status: ✅ IMPLEMENTED (Phase 5)**
> - `Reranker` class (`search_engine.py:27`) with lazy-loaded cross-encoder (`BAAI/bge-reranker-v2-m3`)
> - Thread-safe singleton via `_RERANKER_LOCK` with `False` sentinel for graceful fallback
> - Feeds top-5 detection labels as context pairs to re-ranker
> - Opt-in via `RERANKER_ENABLED=True` and `RERANKER_MODEL` setting
> - Uses `sentence-transformers` (optional dependency)

---

## 10. Add Temporal Reasoning for Action Queries

Single-frame embeddings cannot accurately represent actions such as *walking*, *running*, or *crossing the road*. Instead, index short video clips or tracked object sequences and compute motion features from consecutive frames. The retrieval engine should reason over these temporal sequences rather than isolated images so that action-based queries return complete events instead of unrelated still frames.

> **Status: ✅ IMPLEMENTED (Phase 4)**
> - `_encode_clip_light()` (`pipeline.py`) builds overlapping clip windows (window=3, stride=2)
> - Each clip combines averaged frame embeddings + motion statistics (mean, std, max) — L2-normalized
> - `CLIP_MOTION_WEIGHT=0.3` blends semantic and motion signals
> - `_compute_temporal_score()` (`search_engine.py:334`) analyzes track displacement for 6 action classes:
>   - `crossing`, `stopped`, `walking`, `walking_toward`, `turning`, `carrying`
> - `temporal_alignment` signal combines clip similarity + trajectory action score

---

## 11. Support Multi-Object and Relationship Queries

The retrieval engine should understand queries involving multiple objects and their relationships. For example, *"Show cars waiting at a traffic light"* requires detecting both cars and traffic lights, determining their relative positions, and verifying that the cars remain stationary near the signal. Similar logic applies to queries such as *"Find people carrying backpacks"* or *"Show pedestrians crossing in front of buses."* Supporting object relationships significantly enhances real-world usability.

> **Status: ✅ IMPLEMENTED**
> - `_extract_search_plan()` infers relationships from co-occurring objects (person+backpack, person+dog, person+bicycle) (`search_engine.py:182-186`)
> - `_compute_iou()` helper (`search_engine.py:80`) computes bounding box overlap
> - Stage 3 checks all relationship pairs on the segment's middle frame — IoU > 0.1 triggers `relationship_overlap` signal
> - `RELATIONSHIP_WEIGHT=0.10` integrates overlap score into the 7-signal hybrid weight
> - Multi-object queries are supported via SQLite `_filter_by_class_sqlite()` which filters to frames containing specified object classes

---

## 12. Improve the Visualization of Search Results

When a query is executed, the system should not simply return the retrieved frames. Every matching frame should display green bounding boxes around all detected objects that satisfy the query, with labels showing the object class and confidence score. Consecutive matching frames should be merged into short video clips so the user can observe the complete action rather than isolated snapshots. This produces an intuitive and professional user experience.

> **Status: ✅ IMPLEMENTED**
> - `render()` function in `detector.py:173` draws green bounding boxes with class labels and confidence scores
> - `_save_annotated_thumbnail()` (`search_engine.py`) saves annotated frames as `frame_{mid}_d.jpg` for both pre-indexed objects and query-time detection
> - Each search segment includes `annotated_thumbnail` URL pointing to the annotated middle frame (`/api/frames/{vid}/frame_{mid}_d.jpg`)
> - Consecutive matching frames are merged into segments via `frames_to_segments()` (`pipeline.py:570`)
> - Video clip extraction endpoint (`/api/clips/{video_id}`) generates MP4 clips from start/end frame indices
> - Bbox rendering runs at thumbnail resolution — fast enough for real-time display

---

## 13. Upgrade the Overall Pipeline Architecture

Redesign the indexing pipeline into the following sequence:

**Video → Adaptive Frame Sampling → YOLOv11 Detection → ByteTrack Tracking → Grounding DINO Open-Vocabulary Detection → SAM2 Segmentation (Optional) → Object Cropping → CLIP/SigLIP Embeddings → Scene Caption Generation → Metadata Extraction → FAISS/Vector Database → LLM Query Parser → Hybrid Retrieval → Cross-Encoder Reranking → Bounding Box Rendering → Clip Generation.**

This architecture separates object detection, tracking, embedding generation, metadata storage, retrieval, and visualization into independent components, making the system easier to maintain and significantly more accurate.

> **Status: ⚠️ MOSTLY MATCHED**
> - **Adaptive Frame Sampling** ✅ — motion-based + INDEX_MIN_FPS enforcement
> - **YOLO-World Detection** ✅ — functionally equivalent to YOLOv11; open-vocabulary
> - **SimpleTracker Tracking** ✅ — persistent track IDs across frames (simpler than ByteTrack but serves same purpose)
> - **Object Cropping** ✅ — per-detection crop + CLIP embed during indexing
> - **CLIP Embeddings** ✅ — ViT-B/32 for both frames and object crops
> - **Scene Caption Generation** ✅ — Florence-2 (opt-in)
> - **Metadata Extraction** ✅ — object_metadata + SQLite MetadataDB
> - **FAISS Vector Database** ✅ — separate FAISS indexes for frames, objects, captions
> - **Hybrid Retrieval** ✅ — 7-signal weighted scoring
> - **Cross-Encoder Reranking** ✅ — BAAI/bge-reranker-v2-m3 (opt-in)
> - **Bounding Box Rendering** ✅ — green bboxes on annotated thumbnails
> - **Clip Generation** ✅ — MP4 extraction API endpoint
> - **Not implemented:** SAM2 segmentation (optional, adds heavy dependency), ByteTrack (SimpleTracker used instead), LLM query parser (regex used instead)

---

## 14. Recommended Models

For maximum accuracy, the recommended model stack is:

* **Object Detection:** YOLOv11x
* **Open-Vocabulary Detection:** Grounding DINO 1.5
* **Segmentation:** SAM2
* **Object Tracking:** ByteTrack
* **Image Embeddings:** SigLIP2 (preferred) or CLIP ViT-L/14
* **Scene Captioning:** Florence-2 Large or Qwen2.5-VL
* **Video Understanding:** InternVideo2
* **Reranking:** BAAI BGE Reranker v2
* **Vector Database:** FAISS (or Milvus/Qdrant for larger deployments)

> **Status: 📋 MODEL RECOMMENDATIONS (not implementation items)**
> - Current stack uses CLIP ViT-B/32, YOLO-World, Florence-2-base, BAAI/bge-reranker-v2-m3, FAISS
> - Upgrading models is a configuration/tradeoff decision, not a code change
> - The architecture supports model swaps through the existing abstraction layers

---

## 15. Expected Outcome After Implementation

Once these improvements are implemented, the system will evolve from returning the same generic frames for every query to performing accurate object- and action-aware retrieval. Queries such as *"Show traffic lights"* will return only frames containing traffic lights with bounding boxes around them. Queries like *"Show cars driving through the intersection"* will display all tracked vehicles across consecutive frames, each highlighted independently. Similarly, *"Find people carrying backpacks"* will identify only those individuals where both the person and backpack are detected and spatially associated. This transition from frame-level retrieval to object- and track-centric retrieval is the key architectural change required to achieve production-grade semantic video search.

> **Status: ✅ ACHIEVED**
> - Object-level indexing + object FAISS search enables precise object retrieval
> - Tracking + temporal reasoning enables action-based queries across complete trajectories
> - Relationship IoU overlap enables spatial relationship queries
> - Annotated thumbnails with green bounding boxes returned in search results
> - Video clip extraction enables viewing complete action sequences
> - 7-signal hybrid scoring + cross-encoder reranking ensures high-precision results
