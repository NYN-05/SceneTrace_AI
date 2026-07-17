# Industry-Level Video Retrieval — Technical Upgrade Guide

> **Goal:** Transform SceneTrace AI from CLIP-only frame retrieval into a full **Visual Grounded Video Retrieval** pipeline that answers object-specific, action-specific, and relational queries with production-grade accuracy.

---

## Table of Contents

1. [Current Architecture Assessment](#1-current-architecture-assessment)
2. [Phase 1 — Fix the Foundation](#2-phase-1--fix-the-foundation)
3. [Phase 2 — Object Intelligence](#3-phase-2--object-intelligence)
4. [Phase 3 — Semantic Understanding](#4-phase-3--semantic-understanding)
5. [Phase 4 — Temporal Reasoning](#5-phase-4--temporal-reasoning)
6. [Phase 5 — Hybrid Retrieval &amp; Reranking](#6-phase-5--hybrid-retrieval--reranking)
7. [Target Pipeline](#7-target-pipeline)
8. [Query Breakdown Examples](#8-query-breakdown-examples)
9. [Recommended Models](#9-recommended-models)
10. [Expected Improvements](#10-expected-improvements)
11. [Dependency &amp; Cost Analysis](#11-dependency--cost-analysis)

---

## 1. Current Architecture Assessment

### What We Have Now

```
Video
  │
Adaptive motion sampling (stride=3, 160x90, top 5%)
  │
CLIP Image Encoder (ViT-B/32, batch=32)
  │
FAISS IVFFlat index
  │
User query → CLIP Text Encoder → NN Search → Top-K
  │
YOLO-World-L post-search detection on candidate frames only
  │
Weighted scoring (55% CLIP + 45% object)
  │
Return segments
```

### Root Cause of "Same Frames for Every Query"

| Problem                                | Why It Happens                                                                     | Severity    |
| -------------------------------------- | ---------------------------------------------------------------------------------- | ----------- |
| **CLIP encodes the whole frame** | Query for "traffic light" and "car" both match the same scene vector               | 🔴 Critical |
| **Detection runs after search**  | Objects never enter the index — only frame embeddings are compared                | 🔴 Critical |
| **No object tracking**           | Each frame is independent — "show the white car" returns random frames of any car | 🔴 Critical |
| **No temporal embeddings**       | Action queries like "crossing road" can't be expressed in a single frame           | 🟡 Major    |
| **No hybrid scoring**            | Pure embedding similarity misses structured filters (class, motion, location)      | 🟡 Major    |

---

## 2. Phase 1 — Fix the Foundation

> **Priority:** ⭐⭐⭐⭐⭐ | **Effort:** Medium | **Impact:** High

### Goal

Replace the frame-level-only index with an **object-aware index** that stores per-object embeddings alongside per-frame embeddings.

### Implementation

#### 2.1 Run Detection During Indexing

Currently YOLO-World runs only during search on candidate frames. Move it into the indexing pipeline:

```
For each keyframe:
  1. Run YOLO-World-L detection
  2. For each detected object:
     a. Crop the bounding box from the frame
     b. Encode the crop through CLIP → object_embedding
     c. Store {frame_idx, bbox, class, confidence, object_embedding}
  3. Also encode the full frame → scene_embedding (keep for backward compat)
```

**Code changes in `pipeline.py`:**

```python
from detector import detect  # returns list of {bbox, label, score}

def _index_frame_objects(frame: np.ndarray, frame_idx: int, video_id: str):
    dets = detect(frame, settings.INDEX_DETECTION_PROMPTS)
    objects = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_embed = embed_image(crop)  # CLIP on crop
        objects.append({
            "frame_idx": frame_idx,
            "bbox": d["bbox"],
            "label": d["label"],
            "score": d["score"],
            "embedding": crop_embed.tolist(),
        })
    return objects
```

**New `pipeline.py` changes:**

- Add `INDEX_DETECTION_PROMPTS` to config (default: common categories)
- Add `embed_image()` function for single-image CLIP encoding
- Add `object_index` alongside the existing `embedding_index` in `VideoIndex`
- Store objects in `index.json` under `"objects"` key per frame

#### 2.2 Normalize All Embeddings

Ensure all embeddings (frame, object, text query) are **L2-normalized** before indexing and search. FAISS `IndexFlatIP` and `IndexIVFFlat` both use inner product, and normalization ensures cosine similarity behavior.

```python
def normalize(emb: np.ndarray) -> np.ndarray:
    return emb / np.linalg.norm(emb, axis=-1, keepdims=True)
```

#### 2.3 Cache FAISS in Memory

Already done — FAISS index is built once during `index_video` and loaded on demand. Verify no per-query rebuild path exists.

#### 2.4 Increase Indexing Rate

Current: **adaptive ~1-5 FPS** (depends on motion). Add a configurable minimum FPS:

```python
# config.py
INDEX_MIN_FPS: int = int(os.getenv("INDEX_MIN_FPS", "5"))
```

In pipeline, after motion sampling, ensure at least `INDEX_MIN_FPS` frames per second:

```python
if len(selected_indices) < total_frames / video_fps * settings.INDEX_MIN_FPS:
    # add evenly-spaced frames to meet minimum
    needed = int(total_frames / video_fps * settings.INDEX_MIN_FPS)
    uniform = np.linspace(0, total_frames - 1, needed, dtype=int)
    selected_indices = np.union1d(selected_indices, uniform)
```

#### 2.5 Config Changes

```python
# config.py additions
INDEX_DETECTION_PROMPTS: str = os.getenv("INDEX_DETECTION_PROMPTS",
    "person, car, bicycle, motorcycle, bus, truck, traffic light, fire hydrant, "
    "stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, "
    "elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, "
    "frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, "
    "skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, "
    "spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, "
    "pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, "
    "tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, "
    "sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, "
    "toothbrush")
INDEX_MIN_FPS: int = int(os.getenv("INDEX_MIN_FPS", "5"))
INDEX_OBJECT_CONFIDENCE: float = float(os.getenv("INDEX_OBJECT_CONFIDENCE", "0.3"))
```

### Acceptance Criteria

- [ ] `index.json` contains `"objects"` key with per-frame object entries
- [ ] Each object has `{bbox, label, score, embedding}`
- [ ] Query for "car" returns different results than "traffic light"
- [ ] Indexing time increase is within 2x of current (object detection + CLIP crop is the added cost)

---

## 3. Phase 2 — Object Intelligence

> **Priority:** ⭐⭐⭐⭐⭐ | **Effort:** High | **Impact:** Very High

### Goal

Add **multi-object tracking** across frames so queries return entire trajectories (not scattered single frames).

### Implementation

#### 3.1 Integrate ByteTrack

ByteTrack is a simple, high-performance tracker that assigns consistent IDs to objects across frames using IoU + confidence-based association.

**Installation:**

```bash
pip install bytetrack
# OR use the standalone implementation:
# pip install git+https://github.com/ifzhang/ByteTrack.git
```

**Integration in pipeline:**

```python
from bytetrack import BYTETracker

class ObjectTracker:
    def __init__(self):
        self.tracker = BYTETracker(
            track_thresh=0.5,   # detection confidence threshold
            track_buffer=30,    # frames to keep a lost track alive
            match_thresh=0.8,   # IoU threshold for matching
        )

    def update(self, detections: list, frame_idx: int):
        """
        detections: list of [x1, y1, x2, y2, score, class_id]
        returns: list of {track_id, bbox, class_id, score}
        """
        if not detections:
            self.tracker.update(None)
            return []
        # ByteTrack expects format: [[x1,y1,x2,y2,score,class_id], ...]
        tracks = self.tracker.update(np.array(detections))
        return [{
            "track_id": int(t.track_id),
            "bbox": [int(t.x1), int(t.y1), int(t.x2), int(t.y2)],
            "class_id": int(t.class_id),
            "score": float(t.score),
            "frame_idx": frame_idx,
        } for t in tracks]
```

**Usage in indexing:**

```python
tracker = ObjectTracker()
frame_detections = []  # accumulate across frames

for frame_idx in selected_indices:
    frame = read_frame(frame_idx)
    dets = detect(frame, settings.INDEX_DETECTION_PROMPTS)
    # Format for ByteTrack
    dets_formatted = [
        [d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
         d["score"], class_name_to_id(d["label"])]
        for d in dets
    ]
    tracks = tracker.update(dets_formatted, frame_idx)
    # Store tracks with frame info
    save_tracks(video_id, frame_idx, tracks)
```

#### 3.2 Metadata Schema

```json
{
  "tracks": {
    "7": {
      "class": "car",
      "confidence": 0.92,
      "frames": [
        {"frame_idx": 100, "bbox": [120, 300, 200, 380]},
        {"frame_idx": 105, "bbox": [135, 298, 215, 382]},
        {"frame_idx": 110, "bbox": [150, 295, 230, 385]}
      ],
      "trajectory": {
        "start_frame": 100,
        "end_frame": 350,
        "total_frames": 50,
        "avg_confidence": 0.88,
        "direction": "left-to-right",
        "speed_px_per_frame": 3.2
      }
    }
  }
}
```

#### 3.3 Multi-Object Search

When a user queries "cars", return **all tracked instances** with their full trajectories:

```python
def search_objects(query: str, video_id: str) -> list:
    text_emb = clip_text(query)
    # Search object embeddings in FAISS
    scores, indices = object_index.search(text_emb, k=100)
    # Get unique track IDs from results
    track_ids = set()
    for idx in indices[0]:
        obj = object_store[idx]
        track_ids.add(obj["track_id"])
    # Return full trajectories for matching tracks
    return [load_track(video_id, tid) for tid in track_ids]
```

**Search result format:**

```json
{
  "track_id": 7,
  "class": "car",
  "confidence": 0.92,
  "frames": [
    {"frame_idx": 100, "timestamp": "00:03.33", "bbox": [120, 300, 200, 380]},
    {"frame_idx": 105, "timestamp": "00:03.50", "bbox": [135, 298, 215, 382]},
    ...
  ],
  "clip_url": "/api/clips/{id}?track=7",
  "thumbnail": "/api/frames/{id}/frame_100_d.jpg"
}
```

#### 3.4 Relationship Queries

For "people carrying backpacks" — find overlapping bboxes of two different classes on the same frame:

```python
def search_relationship(obj_a: str, obj_b: str, video_id: str,
                        iou_threshold: float = 0.1):
    """Find frames where obj_a and obj_b overlap (e.g., person + backpack)."""
    a_objs = get_objects_by_class(video_id, obj_a)
    b_objs = get_objects_by_class(video_id, obj_b)
    results = []
    for a in a_objs:
        for b in b_objs:
            if a["frame_idx"] == b["frame_idx"]:
                iou = compute_iou(a["bbox"], b["bbox"])
                if iou > iou_threshold:
                    results.append({
                        "frame_idx": a["frame_idx"],
                        "obj_a": a,
                        "obj_b": b,
                        "iou": iou,
                    })
    return results
```

### Acceptance Criteria

- [ ] Objects have consistent track IDs across frames
- [ ] "Show the white car" returns a single trajectory (not random frames of different cars)
- [ ] "People carrying backpacks" returns frames with person+backpack overlap
- [ ] "Cars crossing intersection" returns multiple track trajectories

---

## 4. Phase 3 — Semantic Understanding

> **Priority:** ⭐⭐⭐⭐☆ | **Effort:** High | **Impact:** High

### Goal

Add **scene captioning** and **LLM-based query parsing** so the system understands natural language intent rather than doing raw embedding comparison.

### Implementation

#### 4.1 Scene Caption Generation

For each keyframe, generate a caption using a vision-language model:

```python
# Install: pip install transformers
from transformers import AutoModelForCausalLM, AutoProcessor

class Captioner:
    def __init__(self, model_name="microsoft/Florence-2-large"):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device).eval()

    @torch.inference_mode()
    def caption(self, image: np.ndarray) -> str:
        prompt = "<MORE_DETAILED_CAPTURE>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=512, num_beams=3)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

# Alternative lighter option: BLIP-2
# pip install transformers
# model = AutoModelForVision2Seq.from_pretrained("Salesforce/blip2-flan-t5-xl")
```

**Caption index structure:**

```json
{
  "frame_idx": 202,
  "caption": "A white sedan drives through an intersection while pedestrians wait at a crosswalk. Traffic lights show green. Buildings line both sides of the street.",
  "caption_embedding": [0.012, -0.034, ...]  # CLIP text embedding of caption
}
```

**Search augmentation:**

```python
# Hybrid: scene embedding + caption embedding + object match
caption_emb = clip_text(caption)
caption_sim = cosine_similarity(query_emb, caption_emb)
final_score = 0.3 * clip_sim + 0.3 * caption_sim + 0.4 * object_match
```

#### 4.2 LLM Query Parser

Use a small, fast LLM to parse natural language queries into structured search plans:

```python
from transformers import pipeline

class QueryParser:
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct"):
        self.pipe = pipeline("text-generation", model=model_name, device=device)

    def parse(self, query: str) -> dict:
        prompt = f"""You are a video search query parser. Extract structured fields from the user query.

Query: "{query}"

Return JSON with these fields if present:
- objects: list of objects to detect (e.g., ["car", "person"])
- actions: list of actions (e.g., ["crossing", "parking"])
- attributes: dict of object attributes (e.g., {{"car": "white"}})
- location: spatial location if specified (e.g., "intersection", "side door")
- min_objects: minimum objects that must be present
- motion: motion pattern if specified (e.g., "fast", "stopped")
- confidence_threshold: minimum confidence (0-1)

JSON:"""
        result = self.pipe(prompt, max_new_tokens=256, temperature=0.1)[0]["generated_text"]
        return json.loads(result.split("JSON:")[-1].strip())

# Example
parser = QueryParser()
plan = parser.parse("Show people carrying backpacks near the entrance")
# Returns:
# {
#   "objects": ["person", "backpack"],
#   "attributes": {},
#   "location": "entrance",
#   "relationships": [{"obj_a": "person", "obj_b": "backpack", "type": "carrying"}],
#   "motion": None
# }
```

**Search plan execution:**

```python
def execute_search(plan: dict, video_id: str) -> list:
    # 1. For each object, search object embeddings
    results_by_object = {}
    for obj in plan.get("objects", []):
        results_by_object[obj] = search_objects(obj, video_id)

    # 2. If relationship (e.g., person + backpack overlap)
    if "relationships" in plan:
        for rel in plan["relationships"]:
            rel_results = search_relationship(rel["obj_a"], rel["obj_b"], video_id)
            # Filter by location if specified
            if "location" in plan:
                rel_results = filter_by_location(rel_results, plan["location"])

    # 3. If attributes (e.g., "white car")
    for obj, attr in plan.get("attributes", {}).items():
        results_by_object[obj] = filter_by_attribute(
            results_by_object[obj], attr
        )

    # 4. If motion (e.g., "crossing", "fast")
    if plan.get("motion"):
        for obj in results_by_object:
            results_by_object[obj] = filter_by_motion(
                results_by_object[obj], plan["motion"]
            )

    return consolidate_results(results_by_object)
```

#### 4.3 Lighter Alternative — Regex + Embedding Approach

If LLM latency is too high, use a rule-based parser for common patterns:

```python
import re

def parse_query_simple(query: str) -> dict:
    query_lower = query.lower()
    objects = []
    attributes = {}
    location = None

    # Object detection patterns
    patterns = {
        "person": r"\b(people?|person|pedestrian|man|woman|child)\b",
        "car": r"\b(car|vehicle|sedan|suv|truck|van)\b",
        "backpack": r"\b(backpack|bag|backpack|rucksack)\b",
        "traffic_light": r"\b(traffic\s*light|signal|stoplight)\b",
    }

    for obj, pattern in patterns.items():
        if re.search(pattern, query_lower):
            objects.append(obj)

    # Attribute extraction
    attr_patterns = [
        (r"(\w+)\s+car", "car"),           # "white car"
        (r"(\w+)\s+(person|man|woman)", "person"),
    ]
    for pattern, obj in attr_patterns:
        m = re.search(pattern, query_lower)
        if m:
            attributes[obj] = m.group(1)

    # Location extraction
    location_patterns = [
        r"(near|at|by|beside)\s+the\s+(\w+)",
        r"(entrance|exit|door|gate|intersection|crosswalk)",
    ]
    for p in location_patterns:
        m = re.search(p, query_lower)
        if m:
            location = m.group(0)

    return {
        "objects": objects,
        "attributes": attributes,
        "location": location,
        "motion": None,
    }
```

### Acceptance Criteria

- [ ] Keyframes have generated captions stored in index
- [ ] Caption similarity contributes to hybrid scoring
- [ ] "White car" correctly filters by attribute (not just class)
- [ ] "People with backpacks" correctly identifies overlapping objects
- [ ] Query parsing takes < 500ms (LLM) or < 10ms (regex)

---

## 5. Phase 4 — Temporal Reasoning

> **Priority:** ⭐⭐⭐⭐⭐ | **Effort:** High | **Impact:** Very High

### Goal

Answer **action queries** ("crossing the road", "walking toward camera") by indexing short video clips rather than isolated frames and using motion features + track trajectories.

### Implementation

#### 5.1 Clip-Level Indexing

Instead of embedding single frames, embed **short video clips** (16-32 frames):

```python
class ClipEncoder:
    def __init__(self):
        # InternVideo2 or VideoMAE for video understanding
        self.model = AutoModel.from_pretrained("OpenGVLab/InternVideo2-Stage2_1B-224p")
        self.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVideo2-Stage2_1B-224p")

    @torch.inference_mode()
    def encode_clip(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames: list of 16-32 np.ndarray frames (H,W,3)"""
        inputs = self.processor(videos=[frames], return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().numpy()
```

**Alternative (lighter):** Average frame embeddings + motion features:

```python
def encode_clip_light(frames: list[np.ndarray]) -> np.ndarray:
    frame_embs = [clip_image(f) for f in frames]
    avg_emb = np.mean(frame_embs, axis=0)

    # Motion features: optical flow between consecutive frames
    motions = []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        motions.append(np.mean(magnitude))

    motion_feat = np.array([np.mean(motions), np.std(motions), np.max(motions)])
    return np.concatenate([avg_emb, motion_feat])
```

#### 5.2 Action Query Handling

Use trajectory analysis to answer action queries:

```python
def filter_by_action(track: dict, action: str) -> bool:
    if action == "crossing":
        # Object moved from one side of frame to the other
        traj = track["trajectory"]
        start_x = traj["bboxes"][0][0]
        end_x = traj["bboxes"][-1][0]
        frame_width = traj.get("frame_width", 640)
        # Crossed more than 50% of frame width
        return abs(end_x - start_x) > frame_width * 0.5

    elif action == "walking_toward":
        # Object size increased over time (approaching camera)
        sizes = [(b[2]-b[0]) * (b[3]-b[1]) for b in traj["bboxes"]]
        return sizes[-1] > sizes[0] * 1.3

    elif action == "stopped":
        # Object barely moved for N frames
        displacements = [abs(traj["bboxes"][i][0] - traj["bboxes"][i+1][0])
                        for i in range(len(traj["bboxes"])-1)]
        return np.mean(displacements) < 5  # pixels per frame

    elif action == "turning":
        # Significant direction change mid-trajectory
        vec1 = np.array(traj["centroids"][len//2]) - np.array(traj["centroids"][0])
        vec2 = np.array(traj["centroids"][-1]) - np.array(traj["centroids"][len//2])
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        return angle > np.pi / 4  # > 45 degrees

    return False
```

#### 5.3 Temporal Segmentation

Group tracks into temporal segments for action recognition:

```python
def segment_by_action(tracks: list, action: str, video_fps: int) -> list:
    """Return continuous segments where the action occurs."""
    segments = []
    current_segment = []
    for track in tracks:
        if filter_by_action(track, action):
            if not current_segment:
                current_segment = [track]
            elif track["frame_idx"] - current_segment[-1]["frame_idx"] <= video_fps:
                current_segment.append(track)
            else:
                segments.append(current_segment)
                current_segment = [track]
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    if current_segment:
        segments.append(current_segment)
    return segments
```

### Acceptance Criteria

- [ ] "Cars crossing intersection" returns only cars that traverse the frame
- [ ] "Person walking toward camera" returns tracks with increasing bbox size
- [ ] "Stopped vehicles" returns stationary cars, not moving ones
- [ ] Clip embeddings improve action query accuracy over single-frame

---

## 6. Phase 5 — Hybrid Retrieval & Reranking

> **Priority:** ⭐⭐⭐⭐⭐ | **Effort:** Medium | **Impact:** Very High

### Goal

Replace single-stage FAISS search with a **multi-stage pipeline**: broad retrieval → structured filtering → cross-encoder reranking.

### Implementation

#### 6.1 Hybrid Scoring Formula

```python
def hybrid_score(
    query_emb: np.ndarray,
    frame_data: dict,
    object_matches: list,
    track_data: dict,
    caption_emb: np.ndarray = None,
    weights: dict = None
) -> float:
    w = weights or {
        "clip_semantic": 0.20,
        "caption_similarity": 0.20,
        "object_match": 0.25,
        "motion_match": 0.10,
        "track_consistency": 0.15,
        "temporal_alignment": 0.10,
    }

    clip_sim = cosine_similarity(query_emb, frame_data["embedding"])
    caption_sim = cosine_similarity(query_emb, caption_emb) if caption_emb is not None else 0
    obj_score = max([m["score"] for m in object_matches]) if object_matches else 0
    motion = frame_data.get("motion_score", 0)
    track_c = track_data.get("consistency", 0) if track_data else 0
    temporal = frame_data.get("temporal_score", 0)

    return (
        w["clip_semantic"] * clip_sim +
        w["caption_similarity"] * caption_sim +
        w["object_match"] * obj_score +
        w["motion_match"] * motion +
        w["track_consistency"] * track_c +
        w["temporal_alignment"] * temporal
    )
```

#### 6.2 Multi-Stage Search Pipeline

```
User Query
  │
  ▼
┌─────────────────────────┐
│ Stage 1: Broad Retrieval │  Top-100 via FAISS on object embeddings
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ Stage 2: Metadata Filter │  Filter by class, track_id, location, motion
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ Stage 3: Hybrid Score    │  Compute weighted score per candidate
└─────────────────────────┘
  │
  ▼
┌─────────────────────────┐
│ Stage 4: Cross-Encoder   │  Rerank top-20 → top-5
└─────────────────────────┘
  │
  ▼
Return final results
```

#### 6.3 Cross-Encoder Reranking

Install:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list:
        pairs = []
        for c in candidates:
            # Create text representation of the candidate
            text = f"Frame {c['frame_idx']}: {c['caption']} Objects: {', '.join(c['object_labels'])}"
            pairs.append((query, text))

        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, s in ranked[:top_k]]
```

#### 6.4 Structured Metadata Database

Replace in-memory JSON with queryable structure. For small scale (<100K objects), use SQLite:

```python
import sqlite3

class MetadataDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id INTEGER PRIMARY KEY,
                video_id TEXT,
                frame_idx INTEGER,
                track_id INTEGER,
                class TEXT,
                confidence REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                timestamp REAL,
                motion_score REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                video_id TEXT,
                frame_idx INTEGER,
                caption TEXT,
                motion_score REAL,
                scene_embedding BLOB
            )
        """)
        self.conn.execute("""
            CREATE INDEX idx_objects_class ON objects(class)
        """)
        self.conn.execute("""
            CREATE INDEX idx_objects_track ON objects(track_id)
        """)

    def query_objects(self, class_name: str = None, track_id: int = None,
                      min_confidence: float = 0.0) -> list:
        conditions = []
        params = []
        if class_name:
            conditions.append("class = ?")
            params.append(class_name)
        if track_id is not None:
            conditions.append("track_id = ?")
            params.append(track_id)
        conditions.append("confidence >= ?")
        params.append(min_confidence)

        sql = "SELECT * FROM objects WHERE " + " AND ".join(conditions)
        return self.conn.execute(sql, params).fetchall()
```

### Acceptance Criteria

- [ ] Hybrid scoring combines all signals with configurable weights
- [ ] Cross-encoder reranking improves top-5 relevance vs. raw FAISS
- [ ] Metadata filters (class, track_id, confidence) narrow results correctly
- [ ] End-to-end query latency stays under 10s for long videos

---

## 7. Target Pipeline

After all 5 phases, the full pipeline becomes:

```
Video
  │
Decode at INDEX_MIN_FPS (5 FPS)
  │
├── Adaptive motion sampling (existing)
│
├── YOLO-World-L object detection (per frame)  ← Phase 1
│   │
│   ├── Crop each object
│   ├── CLIP encode each crop → object_embedding  ← Phase 1
│   └── ByteTrack → track_id assignment  ← Phase 2
│
├── Scene captioning via Florence-2 (per keyframe)  ← Phase 3
│   └── CLIP encode caption → caption_embedding
│
├── Clip-level encoding (16-frame windows)  ← Phase 4
│   └── Motion features + average frame embeddings
│
├── SQLite metadata DB  ← Phase 5
│   └── objects, frames, tracks, captions
│
├── FAISS index:
│   ├── scene_embeddings (frame level)
│   ├── object_embeddings (per object)
│   ├── caption_embeddings
│   └── clip_embeddings (16-frame windows)
│
└── User query
    │
    ├── LLM Query Parser → structured plan  ← Phase 3
    ├── Stage 1: FAISS broad retrieval (top-100)
    ├── Stage 2: Metadata filter (class, track, location)
    ├── Stage 3: Hybrid scoring (6 signals)
    ├── Stage 4: Cross-encoder reranking (top-20 → top-5)  ← Phase 5
    │
    └── Return:
        ├── Track trajectories with full frame lists  ← Phase 2
        ├── Bounding boxes on every frame
        ├── Annotated thumbnails
        ├── Downloadable clips
        └── Score breakdown
```

---

## 8. Query Breakdown Examples

### Query: "Show traffic lights"

```
LLM Parse → {objects: ["traffic light"], attributes: {}, location: None}

1. FAISS: search object_embeddings for "traffic light" → top-100
2. Filter: class == "traffic light"
3. Deduplicate: group by track_id
4. Return: every frame containing a traffic light with bboxes
```

### Query: "Find people carrying backpacks"

```
LLM Parse → {objects: ["person", "backpack"], relationships: [{obj_a: "person", obj_b: "backpack"}]}

1. FAISS: search object_embeddings for "person" → top-100
2. FAISS: search object_embeddings for "backpack" → top-100
3. Join on frame_idx where person AND backpack present
4. Spatial filter: IoU(person_bbox, backpack_bbox) > 0.1
5. Track filter: same track_id for person across frames
6. Return: continuous clip where person+backpack co-occur
```

### Query: "Show cars driving through intersection"

```
LLM Parse → {objects: ["car"], location: "intersection", motion: "crossing"}

1. FAISS: search object_embeddings for "car" → top-100
2. Track: assign track_ids → get full trajectories
3. Motion filter: displacement > 50% frame width (crossing)
4. Location filter: if intersection polygon defined, filter bbox inside polygon
5. Return: all car tracks meeting criteria with full frame sequences
```

### Query: "Show crowded street"

```
LLM Parse → {objects: ["person", "car"], min_objects: {person: 20, car: 10}}

1. FAISS: search object_embeddings for "person" → top-200
2. FAISS: search object_embeddings for "car" → top-200
3. Count per frame: person_count >= 20 AND car_count >= 10
4. Alternative: search captions for "crowded", "busy", "street"
5. Return: frames meeting count threshold with all bboxes
```

### Query: "Show the white car that was parked then drove away"

```
LLM Parse → {
  objects: ["car"],
  attributes: {car: "white"},
  actions: [{action: "stopped", sequence: 1}, {action: "crossing", sequence: 2}]
}

1. FAISS: search for "white car" → tracks with attribute=white
2. Temporal split: for each track, find stopped segment → moving segment
3. Return: clip showing entire sequence (parked → driving away)
```

---

## 9. Recommended Models

| Task                            | Recommended Model    | Size   | VRAM   | Speed     | Alternative (Lighter)              |
| ------------------------------- | -------------------- | ------ | ------ | --------- | ---------------------------------- |
| **Object Detection**      | YOLOv11x             | ~350MB | 4-6 GB | Very Fast | YOLOv11m (~200MB)                  |
| **Open-Vocab Detection**  | Grounding DINO 1.5   | ~1.5GB | 4-6 GB | Slow      | YOLO-World-L (already integrated)  |
| **Segmentation**          | SAM2                 | ~2.5GB | 6-8 GB | Medium    | SAM2-Tiny (~500MB)                 |
| **Object Tracking**       | ByteTrack            | ~0MB   | 0 GB   | Very Fast | (None needed)                      |
| **Image Embedding**       | SigLIP2 ViT-L/14     | ~1.5GB | 3-4 GB | Fast      | CLIP ViT-B/32 (already integrated) |
| **Caption Generation**    | Florence-2-Large     | ~1.8GB | 4-6 GB | Slow      | BLIP-2 Flan-T5-XL (~3GB)           |
| **Caption Gen (Lighter)** | Florence-2-Base      | ~800MB | 2-3 GB | Medium    |                                    |
| **Video Understanding**   | InternVideo2         | ~2.5GB | 6-8 GB | Medium    | Frame averaging (no extra model)   |
| **Reranking**             | BAAI BGE-Reranker-v2 | ~1.0GB | 2-3 GB | Medium    | Cross-encoder MiniLM (~500MB)      |
| **LLM Query Parser**      | Phi-3-Mini-4K        | ~2.5GB | 4-6 GB | Fast      | Regex parser (no extra model)      |
| **LLM Query (Lighter)**   | Llama-3.2-3B         | ~2.0GB | 3-4 GB | Fast      |                                    |

### Cumulative Memory Budget

| Phase   | Additional Models        | Peak VRAM | Total VRAM |
| ------- | ------------------------ | --------- | ---------- |
| Current | CLIP + YOLO-World-L      | ~2 GB     | ~2 GB      |
| Phase 1 | (none added)             | —        | ~2 GB      |
| Phase 2 | ByteTrack (no GPU)       | 0 GB      | ~2 GB      |
| Phase 3 | Florence-2-L or Phi-3    | ~2-4 GB   | ~4-6 GB    |
| Phase 4 | InternVideo2 or VideoMAE | ~3 GB     | ~5-7 GB    |
| Phase 5 | BGE-Reranker             | ~1 GB     | ~5-8 GB    |

> **VRAM ceiling:** If using a single 8-12 GB GPU, you cannot load all models simultaneously. Solutions:
>
> - **Model swapping:** Load models on demand, unload after use
> - **CPU inference:** Run captioning/LLM on CPU (slower but viable)
> - **Batch processing:** Run captioning offline during indexing, not at query time
> - **Phased deployment:** Implement phases incrementally; test VRAM after each

---

## 10. Expected Improvements

### Per-Phase Accuracy Gains

| Phase             | Metric                                        | Before | After | Improvement    |
| ----------------- | --------------------------------------------- | ------ | ----- | -------------- |
| **Phase 1** | Object-level retrieval precision              | ~30%   | ~65%  | 2x             |
|                   | Distinct results for different object queries | Low    | High  | —             |
| **Phase 2** | Trajectory completeness                       | N/A    | 90%+  | New capability |
|                   | Multi-frame consistency                       | Low    | High  | —             |
| **Phase 3** | Natural language understanding                | ~50%   | ~85%  | 1.7x           |
|                   | Attribute-aware filtering (e.g., "white car") | None   | 80%+  | New capability |
| **Phase 4** | Action query accuracy (crossing, turning)     | ~20%   | ~75%  | 3.7x           |
|                   | Temporal segmentation quality                 | Low    | High  | —             |
| **Phase 5** | Overall ranking quality (nDCG@10)             | ~0.4   | ~0.8  | 2x             |
|                   | False positive reduction                      | High   | Low   | —             |

### Final Target Metrics

| Metric                            | Current | Target                                      |
| --------------------------------- | ------- | ------------------------------------------- |
| Object retrieval accuracy         | ~30-40% | **90-97%**                            |
| Natural language understanding    | ~50%    | **90-95%**                            |
| Bounding-box localization         | ~70%    | **95%+**                              |
| Multi-object tracking consistency | N/A     | **90%+**                              |
| Action query accuracy             | ~20%    | **80%+**                              |
| Query latency (v2 search)         | ~5s     | **< 8s** (with reranking)             |
| Indexing speed                    | ~28 FPS | **~20 FPS** (with detection overhead) |
| Frame reduction                   | 97%     | 95% (more frames indexed)                   |

---

## 11. Dependency & Cost Analysis

### New Dependencies

```txt
# Phase 1 — No new dependencies (uses existing YOLO-World)
# Phase 2
git+https://github.com/ifzhang/ByteTrack.git
# Phase 3
transformers>=4.45.0
sentencepiece
accelerate
# Phase 4 (optional)
einops
decord
# Phase 5
sentence-transformers>=3.0.0
```

### Implementation Order (Recommended)

| Step | Phase                                     | Effort | Risk   | Value     | Do First?                      |
| ---- | ----------------------------------------- | ------ | ------ | --------- | ------------------------------ |
| 1    | Phase 1: Object detection during indexing | 2 days | Low    | High      | ✅ Yes D                       |
| 2    | Phase 2: ByteTrack integration            | 2 days | Low    | Very High | ✅ YesD                        |
| 3    | Phase 3: Regex query parser (light)       | 1 day  | Low    | Medium    | ✅ YesD                        |
| 4    | Phase 3: Scene captioning                 | 3 days | Medium | High      | ⬜ After 1-3                   |
| 5    | Phase 3: LLM query parser                 | 2 days | Medium | High      | ⬜ After captioning            |
| 6    | Phase 4: Clip-level indexing              | 4 days | High   | High      | ⬜ Later                       |
| 7    | Phase 4: Action query engine              | 3 days | High   | High      | ⬜ With clip indexing          |
| 8    | Phase 5: Hybrid scoring formula           | 1 day  | Low    | High      | ✅ Yes (combine with step 1) D |
| 9    | Phase 5: Metadata DB                      | 2 days | Medium | Medium    | ⬜ After step 2D               |
| 10   | Phase 5: Cross-encoder reranking          | 1 day  | Low    | High      | ⬜ After hybrid scoring D      |

### Quick Wins (Week 1)

1. **Phase 1 — Object detection during indexing** (2 days)

   - Biggest single improvement: queries for different objects return different results
   - Reuses existing YOLO-World-L code
   - No new model downloads
2. **Phase 2 — ByteTrack** (2 days)

   - Transforms scattered frames into coherent tracks
   - "Show the white car" becomes meaningful
   - Zero GPU cost
3. **Phase 5 — Hybrid scoring** (1 day)

   - Combine existing CLIP + object signals with track consistency
   - Configurable weights → tunable per dataset

**Total week 1:** ~5 days for 3 biggest-impact changes. Expected retrieval accuracy improvement: **30% → ~75%**.

---

*This document provides a complete technical roadmap. Each phase can be implemented independently. Start with Phase 1 for the highest ROI: moving object detection into the indexing pipeline so object-level embeddings become part of the search index.*
