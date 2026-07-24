"""Enhanced search: CLIP + object detection + weighted scoring + explanations.

Multi-stage pipeline (Phase 5):
  Stage 1 — FAISS semantic retrieval
  Stage 2 — SQLite class filter (when class names found in query)
  Stage 3 — 6-signal hybrid scoring with configurable weights
  Stage 4 — Cross-encoder reranker (optional)
"""
import math
import re
import time
import cv2
import numpy as np
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from cachetools import TTLCache
from pipeline import STORAGE, VideoIndex, search_embeddings, frames_to_segments, embed_text, MetadataDB
from config import benchmark, settings

logger = logging.getLogger("scenetrace.search")

_RERANKER_LOCK = threading.Lock()
_RERANKER_MODEL = None

# ── Stage 4: Cross-encoder reranker ──────────────────────────

class Reranker:
    """Optional cross-encoder reranker (Stage 4). Lazy-loaded; graceful fallback."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.RERANKER_MODEL

    def rerank(self, query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
        global _RERANKER_MODEL
        if _RERANKER_MODEL is False:
            return candidates[:top_k] if top_k else candidates
        if _RERANKER_MODEL is None:
            with _RERANKER_LOCK:
                if _RERANKER_MODEL is None:
                    try:
                        from sentence_transformers import CrossEncoder
                        _RERANKER_MODEL = CrossEncoder(self.model_name)
                    except Exception:
                        logger.warning("Failed to load reranker '%s'; reranking disabled", self.model_name)
                        _RERANKER_MODEL = False
                        return candidates[:top_k] if top_k else candidates
        top = top_k or len(candidates)
        pairs = []
        for c in candidates:
            labels = [d.get("label", "") for d in c.get("detections", [])][:5]
            desc = f"objects: {', '.join(labels)}" if labels else "no objects"
            pairs.append((query, f"query: {query} candidate: {desc}"))
        try:
            scores = _RERANKER_MODEL.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:top]]
        except Exception:
            logger.exception("Reranker predict failed; using original order")
            return candidates[:top] if top_k else candidates


# ── Stage 2: SQLite metadata filter ──────────────────────────

_COMMON_CLASSES = {
    "person", "car", "bicycle", "motorcycle", "bus", "truck", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}


def _compute_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_class_names(query: str) -> list[str]:
    """Return object class names found in query text."""
    q = query.lower()
    found = []
    for cls in sorted(_COMMON_CLASSES, key=len, reverse=True):
        if cls in q:
            found.append(cls)
            q = q.replace(cls, "", 1)
    return found


def _filter_by_class_sqlite(vid: str, class_names: list[str],
                            min_confidence: float = 0.0) -> set[int] | None:
    """Return set of frame_idx containing *any* class_name (from SQLite), or None if no DB."""
    db_path = STORAGE / "frames" / vid / "metadata.db"
    if not db_path.exists():
        return None
    try:
        mdb = MetadataDB(db_path)
        frames: set[int] = set()
        for cls in class_names:
            rows = mdb.query_objects(class_name=cls, min_confidence=min_confidence)
            frames.update(r["frame_idx"] for r in rows)
        mdb.close()
        return frames
    except Exception:
        logger.exception("SQLite filter failed for %s", vid)
        return None


# ── Phase 3: Query parser (regex) ─────────────────────────────

_ATTR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(\w+)\s+car\b", re.I), "car"),
    (re.compile(r"(\w+)\s+(person|man|woman|child)\b", re.I), "person"),
    (re.compile(r"(\w+)\s+(truck|van|bus|bicycle|motorcycle)\b", re.I), None),
    (re.compile(r"(\w+)\s+dog\b", re.I), "dog"),
    (re.compile(r"(\w+)\s+cat\b", re.I), "cat"),
]

_LOCATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(near|at|by|beside)\s+the\s+(\w+)", re.I),
    re.compile(r"\b(entrance|exit|door|gate|intersection|crosswalk|sidewalk|driveway|parking)", re.I),
    re.compile(r"\b(street|road|highway|path|corner|stairs|elevator|lobby)\b", re.I),
]

_ACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(walk(ing)?|runs?|running)\b", re.I), "walking"),
    (re.compile(r"\b(driv(ing|es?)|cross(ing|es)?)\b", re.I), "crossing"),
    (re.compile(r"\b(park(ed|ing)?|stop(ped|ing)?|stationary)\b", re.I), "stopped"),
    (re.compile(r"\b(carr(y|ies|ing)|hold(ing)?|with)\b", re.I), "carrying"),
    (re.compile(r"\b(toward|approach(ing)?|coming)\b", re.I), "walking_toward"),
    (re.compile(r"\b(turn(ing|s|ed)?|rotate?|rotating)\b", re.I), "turning"),
]


def _extract_search_plan(query: str) -> dict:
    """Parse query into structured search plan using regex.

    Returns:
        objects: list of detected object class names
        attributes: dict of obj → attribute (e.g. {"car": "white"})
        location: str or None
        actions: list of action types
        relationships: list of {obj_a, obj_b} pairs (e.g. person+backpack co-occurrence)
    """
    q = query.lower()
    objects = list(_extract_class_names(query))
    attributes: dict[str, str] = {}
    location: str | None = None
    actions: list[str] = []
    relationships: list[dict] = []

    for pattern, obj in _ATTR_PATTERNS:
        m = pattern.search(q)
        if m:
            val = m.group(1).lower()
            target = obj or m.group(2).lower()
            if val not in ("a", "an", "the", "my", "your", "some", "this", "that"):
                attributes[target] = val

    for p in _LOCATION_PATTERNS:
        m = p.search(q)
        if m:
            location = m.group(0)
            break

    for pattern, action in _ACTION_PATTERNS:
        if pattern.search(q):
            actions.append(action)

    if "person" in objects and "backpack" in objects:
        relationships.append({"obj_a": "person", "obj_b": "backpack"})
    if "person" in objects and "dog" in objects:
        relationships.append({"obj_a": "person", "obj_b": "dog"})
    if "person" in objects and "bicycle" in objects:
        relationships.append({"obj_a": "person", "obj_b": "bicycle"})

    return {
        "objects": objects,
        "attributes": attributes,
        "location": location,
        "actions": actions,
        "relationships": relationships,
    }


# ── Stage 3 helpers: object / track / motion / caption signals ─

_DET_CACHE: TTLCache = TTLCache(maxsize=settings.DET_CACHE_MAXSIZE, ttl=settings.DET_CACHE_TTL)


def _get_detections_for_segment(seg: dict, query: str) -> list:
    vid, indices = seg["video_id"], seg["frame_indices"]
    mid = indices[len(indices) // 2]
    key = f"{vid}_{mid}"
    if key in _DET_CACHE:
        return _DET_CACHE[key]
    from detector import detect
    img_path = STORAGE / "frames" / vid / f"frame_{mid}.jpg"
    if not img_path.exists():
        _DET_CACHE[key] = []
        return []
    img = cv2.imread(str(img_path))
    if img is None:
        _DET_CACHE[key] = []
        return []
    try:
        dets = detect(img, query, threshold=settings.DETECTION_THRESHOLD)
    except Exception:
        logger.exception("Detection failed for %s", key)
        dets = []
    if dets:
        from detector import render
        annotated = render(img, dets)
        out = STORAGE / "frames" / vid / f"frame_{mid}_d.jpg"
        cv2.imwrite(str(out), annotated)
    _DET_CACHE[key] = dets
    return dets


def _save_annotated_thumbnail(vid: str, frame_idx: int, detections: list[dict]):
    img_path = STORAGE / "frames" / vid / f"frame_{frame_idx}.jpg"
    if not img_path.exists():
        return
    img = cv2.imread(str(img_path))
    if img is None:
        return
    from detector import render
    annotated = render(img, detections)
    out = STORAGE / "frames" / vid / f"frame_{frame_idx}_d.jpg"
    cv2.imwrite(str(out), annotated)


def _search_objects_faiss(query: str, vid: str, idx,
                          top_k: int = 15,
                          sqlite_frame_set: set[int] | None = None,
                          ) -> tuple[dict[int, list[dict]], dict]:
    """Query object FAISS → (frame_map, track_info).

    Optionally filters results to only frames in *sqlite_frame_set* (Stage 2).
    """
    obj_faiss = idx.get_object_faiss_index()
    obj_embs = idx.load_object_embeddings()
    if obj_faiss is None or obj_embs is None or len(obj_embs) == 0:
        return {}, {}
    if len(idx.object_metadata) != len(obj_embs):
        logger.warning("object_metadata (%d) vs obj_embs (%d) mismatch for %s; using min",
                       len(idx.object_metadata), len(obj_embs), vid)
        n_meta = min(len(idx.object_metadata), len(obj_embs))
        obj_meta = idx.object_metadata[:n_meta]
        obj_embs = obj_embs[:n_meta]
    else:
        obj_meta = idx.object_metadata
    q_emb = embed_text([query])
    n = len(obj_embs)
    scores, indices = obj_faiss.search(q_emb, min(top_k * 3, n))
    frame_map: dict[int, list[dict]] = {}
    track_hits: dict[int, list[float]] = {}
    for obj_idx, score in zip(indices[0], scores[0]):
        if obj_idx < 0 or obj_idx >= len(obj_meta):
            continue
        s_val = float(score)
        if not math.isfinite(s_val):
            s_val = 0.0
        meta = obj_meta[obj_idx]
        fi = meta["frame_idx"]
        if sqlite_frame_set is not None and fi not in sqlite_frame_set:
            continue
        tid = meta.get("track_id", -1)
        entry = {
            "bbox": meta["bbox"],
            "label": meta["label"],
            "score": meta["score"],
            "clip_similarity": round(float(s_val), 3),
            "track_id": tid,
        }
        if fi not in frame_map:
            frame_map[fi] = []
        frame_map[fi].append(entry)
        if tid >= 0:
            if tid not in track_hits:
                track_hits[tid] = []
            track_hits[tid].append(float(score))

    track_info: dict = {}
    for tid, sims in track_hits.items():
        full = idx.track_metadata.get(str(tid))
        if full:
            track_info[tid] = {
                "class": full["class"],
                "total_frames": full["total_frames"],
                "avg_confidence": full["avg_confidence"],
                "displacement": full["displacement"],
                "match_score": round(sum(sims) / len(sims), 3),
            }
    return frame_map, track_info


def _compute_motion_score(mid_frame: int, idx) -> float:
    """Return motion_activity score (0-1) for a keyframe, or 0 if unavailable."""
    if not idx.frame_indices or not idx.motion_scores:
        return 0.0
    try:
        pos = idx.frame_indices.index(mid_frame)
        if pos < len(idx.motion_scores):
            return round(float(min(idx.motion_scores[pos], 1.0)), 3)
    except (ValueError, AttributeError, IndexError):
        pass
    return 0.0


def _search_captions_faiss(query: str, idx) -> dict[int, float]:
    """Search caption FAISS → {frame_idx: caption_similarity}."""
    cap_embs = idx.load_caption_embeddings()
    cap_faiss = idx.get_caption_faiss_index()
    if cap_faiss is None or cap_embs is None or len(cap_embs) == 0:
        return {}
    q_emb = embed_text([query])
    n = len(cap_embs)
    scores, indices = cap_faiss.search(q_emb, min(50, n))
    frame_to_sim: dict[int, float] = {}
    for pos, score in zip(indices[0], scores[0]):
        if pos < 0 or pos >= len(idx.frame_indices):
            continue
        fi = idx.frame_indices[pos]
        s = float(score)
        if not (math.isfinite(s) and -1.0 <= s <= 1.0):
            s = 0.0
        s = max(0.0, s)
        if s > 0:
            frame_to_sim[fi] = round(float(s), 3)
    return frame_to_sim


def _search_clip_embeddings(query: str, idx) -> dict[int, float]:
    """Linear search of clip embeddings → {frame_idx: temporal_score}.

    Each clip covers a sliding window of keyframes; the score is assigned
    to every frame in that clip's window. Backward compatible (empty dict
    if no clip embeddings exist).
    """
    clip_embs = idx.load_clip_embeddings()
    if clip_embs is None or len(clip_embs) == 0 or not idx.clip_indices:
        return {}
    q_emb = embed_text([query])
    if clip_embs.shape[1] == 515 and q_emb.shape[1] == 512:
        clip_embs = clip_embs[:, :512].copy()
    sims = np.dot(clip_embs, q_emb.T).flatten()
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=0.0)
    frame_to_score: dict[int, float] = {}
    for clip_frames, sim in zip(idx.clip_indices, sims):
        s = round(float(max(0.0, min(1.0, sim))), 3)
        for fi in clip_frames:
            if s > frame_to_score.get(fi, 0.0):
                frame_to_score[fi] = s
    return frame_to_score


def _compute_temporal_score(seg: dict, track_info: dict, actions: list[str],
                            metadata: dict) -> float:
    """Score a segment's temporal alignment with requested actions using trajectory analysis.

    Returns 0-1 score based on track displacement and frame counts.
    """
    if not actions or not track_info:
        return 0.0

    frame_width = metadata.get("width", 640)
    scores: list[float] = []
    for tid, ti in track_info.items():
        if tid not in seg.get("track_ids", set()):
            continue
        displacement = ti.get("displacement", 0)
        total_frames = ti.get("total_frames", 1)

        for action in actions:
            if action == "crossing":
                ratio = min(displacement / (frame_width * 0.5), 1.0)
                scores.append(ratio)
            elif action == "stopped":
                stillness = max(0.0, 1.0 - displacement / max(frame_width * 0.1, 1))
                scores.append(stillness)
            elif action == "walking":
                ratio = min(displacement / (frame_width * 0.3), 1.0)
                duration = min(total_frames / 5.0, 1.0)
                scores.append(ratio * 0.6 + duration * 0.4)
            elif action == "walking_toward":
                duration = min(total_frames / 5.0, 1.0)
                scores.append(duration * 0.7 + 0.3)
            elif action == "turning":
                ratio = min(displacement / (frame_width * 0.4), 1.0)
                scores.append(ratio * 0.5 + 0.25)
            elif action == "carrying":
                scores.append(0.5)
    return round(max(scores) if scores else 0.0, 3)


def _hybrid_score(seg: dict) -> tuple[float, dict]:
    """Compute 6-signal weighted hybrid score for a segment.

    Signals (roadmap §6.1):
      1. clip_semantic        — CLIP similarity to query text
      2. caption_similarity   — caption-embedding similarity (Phase 3)
      3. object_match          — object-embedding similarity
      4. motion_match          — per-keyframe motion activity
      5. track_consistency     — track persistence / match score
       6. temporal_alignment    — clip-embedding + trajectory analysis for actions (Phase 4)
       7. relationship_overlap  — spatial overlap (IoU) between related objects (Phase 2)
    """
    s = seg.get("semantic_score", 0)
    c = seg.get("caption_similarity", 0)
    o = seg.get("object_score", 0)
    m = seg.get("motion_activity", 0)
    t = seg.get("tracking_consistency", 0)
    ta = seg.get("temporal_alignment", 0)
    ro = seg.get("relationship_overlap", 0)

    weighted = (
        settings.CLIP_WEIGHT * s +
        settings.CAPTION_WEIGHT * c +
        settings.OBJECT_MATCH_WEIGHT * o +
        settings.MOTION_MATCH_WEIGHT * m +
        settings.TRACK_CONSISTENCY_WEIGHT * t +
        settings.TEMPORAL_WEIGHT * ta +
        settings.RELATIONSHIP_WEIGHT * ro
    )
    w = round(weighted, 4)
    breakdown = {
        "clip_semantic": round(s, 3),
        "caption_similarity": round(c, 3),
        "object_match": round(o, 3),
        "motion_activity": round(m, 3),
        "tracking_consistency": round(t, 3),
        "temporal_alignment": round(ta, 3),
        "relationship_overlap": round(ro, 3),
        "weighted_total": w,
    }
    return w, breakdown


# ── Main entry point ─────────────────────────────────────────

def search(query: str, indexes: dict, top_k: int = 5, enable_detection: bool = True) -> dict:
    t0 = time.time()
    if not indexes:
        return {"segments": [], "status": "no_indexes", "query_info": {"raw": query}}

    # ── Stage 2 prep: extract search plan + class names ──
    search_plan = _extract_search_plan(query) if enable_detection else {}
    class_names = search_plan.get("objects", []) if enable_detection else []

    # Expand query with attributes for better embedding matching (e.g. "white car")
    expanded_query = query
    attrs = search_plan.get("attributes", {})
    if attrs:
        expanded_query = query + " " + " ".join(f"{v} {k}" for k, v in attrs.items())

    # ── Stage 1: FAISS semantic retrieval per video ──
    candidates: list[dict] = []
    object_frame_maps: dict[str, dict[int, list[dict]]] = {}
    object_track_info: dict[str, dict] = {}
    caption_frame_sims: dict[str, dict[int, float]] = {}
    clip_frame_sims: dict[str, dict[int, float]] = {}

    vid_list = list(indexes.items())

    # Run all three independent per-video search loops in parallel
    def _search_video_semantic(vid: str, idx: VideoIndex) -> list[dict]:
        embs = idx.load_embeddings()
        if embs is None or len(embs) == 0:
            return []
        faiss_idx = idx.get_faiss_index()
        indices, scores = search_embeddings(expanded_query, faiss_idx, embs, top_k * 3)
        paired = sorted(zip(indices, scores), key=lambda x: idx.frame_indices[x[0]])
        sorted_idx, sorted_sc = zip(*paired) if paired else ([], [])
        segs = frames_to_segments(list(sorted_idx), list(sorted_sc))
        for s in segs:
            orig = s["frame_indices"]
            s["video_id"] = vid
            s["frame_indices"] = [idx.frame_indices[i] for i in orig]
            s["timestamps"] = [idx.timestamps[i] for i in orig]
            s["semantic_score"] = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        segs.sort(key=lambda s: s["semantic_score"], reverse=True)
        return segs[:top_k * 2]

    def _search_video_objects(vid: str, idx: VideoIndex):
        if not enable_detection:
            return {}, {}
        sqlite_frames: set[int] | None = None
        if class_names:
            sqlite_frames = _filter_by_class_sqlite(vid, class_names)
        return _search_objects_faiss(expanded_query, vid, idx, top_k, sqlite_frame_set=sqlite_frames)

    def _search_video_caption(vid: str, idx: VideoIndex) -> dict[int, float]:
        return _search_captions_faiss(expanded_query, idx) or {}

    def _search_video_clip(vid: str, idx: VideoIndex) -> dict[int, float]:
        return _search_clip_embeddings(expanded_query, idx) or {}

    n_vids = len(vid_list)
    if n_vids > 1:
        sem_results: list[list[dict]] = [[] for _ in range(n_vids)]
        obj_maps: list[dict[str, dict[int, list[dict]]]] = [{} for _ in range(n_vids)]
        obj_tracks: list[dict[str, dict]] = [{} for _ in range(n_vids)]
        cap_results: list[dict[int, float]] = [{} for _ in range(n_vids)]
        clip_results: list[dict[int, float]] = [{} for _ in range(n_vids)]

        def _run_sem():
            for i, (vid, idx) in enumerate(vid_list):
                sem_results[i] = _search_video_semantic(vid, idx)
        def _run_obj():
            for i, (vid, idx) in enumerate(vid_list):
                om, ot = _search_video_objects(vid, idx)
                obj_maps[i] = om
                obj_tracks[i] = ot
        def _run_cap():
            for i, (vid, idx) in enumerate(vid_list):
                cap_results[i] = _search_video_caption(vid, idx)
        def _run_clip():
            for i, (vid, idx) in enumerate(vid_list):
                clip_results[i] = _search_video_clip(vid, idx)

        with ThreadPoolExecutor(max_workers=4) as pool:
            fs = [pool.submit(_run_sem), pool.submit(_run_obj), pool.submit(_run_cap), pool.submit(_run_clip)]
            for f in as_completed(fs):
                f.result()

        for i, (vid, _) in enumerate(vid_list):
            candidates.extend(sem_results[i])
            if obj_maps[i]:
                object_frame_maps[vid] = obj_maps[i]
            if obj_tracks[i]:
                object_track_info[vid] = obj_tracks[i]
            if cap_results[i]:
                caption_frame_sims[vid] = cap_results[i]
            if clip_results[i]:
                clip_frame_sims[vid] = clip_results[i]
    else:
        for vid, idx in vid_list:
            candidates.extend(_search_video_semantic(vid, idx))
            if enable_detection:
                om, ot = _search_video_objects(vid, idx)
                if om:
                    object_frame_maps[vid] = om
                if ot:
                    object_track_info[vid] = ot
            cap_sims = _search_video_caption(vid, idx)
            if cap_sims:
                caption_frame_sims[vid] = cap_sims
            clip_sims = _search_video_clip(vid, idx)
            if clip_sims:
                clip_frame_sims[vid] = clip_sims

    # ── Stage 3: compute all 6 signals per candidate (parallel) ──
    def _score_segment(seg: dict) -> dict:
        mid = seg["frame_indices"][len(seg["frame_indices"]) // 2]
        vid = seg["video_id"]

        if enable_detection and vid in object_frame_maps and mid in object_frame_maps[vid]:
            objs = object_frame_maps[vid][mid]
            seg["detections"] = objs
            _save_annotated_thumbnail(vid, mid, objs)
            sims = [o["clip_similarity"] for o in objs]
            seg["object_score"] = round(sum(sims) / len(sims), 4) if sims else 0
            tids = set(o.get("track_id", -1) for o in objs if o.get("track_id", -1) >= 0)
            seg["track_ids"] = tids
            if tids and vid in object_track_info:
                trk_scores = []
                for tid in tids:
                    ti = object_track_info[vid].get(tid)
                    if ti:
                        tf = ti.get("total_frames", 0)
                        norm = min(tf / max(10.0, 1.0), 1.0)
                        trk_scores.append(norm * ti.get("match_score", 0))
                seg["tracking_consistency"] = round(sum(trk_scores) / len(trk_scores), 3) if trk_scores else 0
            else:
                seg["tracking_consistency"] = 0
        elif enable_detection:
            try:
                dets = _get_detections_for_segment(seg, query)
                seg["detections"] = dets
                obj_scores = [d["score"] * 0.5 for d in dets]
                seg["object_score"] = round(sum(obj_scores) / max(len(obj_scores), 1), 4) if obj_scores else 0
                seg["tracking_consistency"] = 0
            except Exception:
                logger.exception("Detection error on %s", vid)
                seg["detections"] = []
                seg["object_score"] = 0
                seg["tracking_consistency"] = 0
        else:
            seg["detections"] = []
            seg["object_score"] = 0
            seg["tracking_consistency"] = 0

        if seg.get("detections"):
            seg["annotated_thumbnail"] = f"/api/frames/{vid}/frame_{mid}_d.jpg"

        relationships = search_plan.get("relationships", [])
        relationship_iou = 0.0
        if relationships and enable_detection and vid in object_frame_maps and mid in object_frame_maps[vid]:
            objs_mid = object_frame_maps[vid][mid]
            for rel in relationships:
                group_a = [o for o in objs_mid if o["label"] == rel["obj_a"]]
                group_b = [o for o in objs_mid if o["label"] == rel["obj_b"]]
                for a in group_a:
                    for b in group_b:
                        iou = _compute_iou(a["bbox"], b["bbox"])
                        if iou > 0.1:
                            relationship_iou = max(relationship_iou, round(iou, 3))
        seg["relationship_overlap"] = relationship_iou

        cap_sim = 0.0
        if vid in caption_frame_sims and mid in caption_frame_sims[vid]:
            cap_sim = caption_frame_sims[vid][mid]
        seg["caption_similarity"] = cap_sim

        temp_align = 0.0
        if vid in clip_frame_sims and mid in clip_frame_sims[vid]:
            temp_align = clip_frame_sims[vid][mid]
        actions = search_plan.get("actions", [])
        if actions and vid in object_track_info:
            seg["track_ids"] = seg.get("track_ids", set())
            traj_score = _compute_temporal_score(
                seg, object_track_info[vid], actions,
                indexes[vid].metadata,
            )
            temp_align = max(temp_align, traj_score)
        seg["temporal_alignment"] = temp_align

        motion = _compute_motion_score(mid, indexes[vid])
        actions = search_plan.get("actions", [])
        if "stopped" in actions:
            motion = 1.0 - motion
        elif "crossing" in actions or "walking" in actions:
            pass
        seg["motion_activity"] = round(motion, 3)

        w, breakdown = _hybrid_score(seg)
        seg["weighted_score"] = w
        seg["score_breakdown"] = breakdown
        seg["avg_score"] = w
        return seg

    n_segs = len(candidates)
    if n_segs > 4:
        with ThreadPoolExecutor(max_workers=min(settings.NUM_WORKERS, n_segs)) as pool:
            candidates = list(pool.map(_score_segment, candidates))
    else:
        candidates = [_score_segment(s) for s in candidates]

    # ── Sort by hybrid score ──
    candidates.sort(key=lambda s: s["weighted_score"], reverse=True)
    segments = candidates[:top_k]

    # ── Stage 4: optional cross-encoder reranking ──
    if settings.RERANKER_ENABLED and enable_detection and len(segments) > 1:
        try:
            rr = Reranker()
            segments = rr.rerank(query, segments, top_k)
        except Exception:
            logger.exception("Reranking failed; using weighted-score order")

    top_score = segments[0]["weighted_score"] if segments else 0
    status = ("high" if top_score > settings.SEARCH_HIGH_THRESHOLD
              else "medium" if top_score > settings.SEARCH_MEDIUM_THRESHOLD
              else "low")

    elapsed = time.time() - t0
    benchmark.record_query(elapsed)

    query_info = {"raw": query, "semantic_query": query}
    if search_plan:
        query_info["search_plan"] = search_plan

    return {
        "segments": segments,
        "status": status,
        "query_info": query_info,
        "query_time": round(elapsed, 2),
    }

def suggest(text: str) -> list[str]:
    suggestions = {
        "person": ["person walking", "person running", "person carrying something", "person entering", "person leaving"],
        "car": ["car driving", "car parked", "car entering", "car leaving"],
        "backpack": ["person with backpack", "backpack on ground", "picking up backpack"],
        "red": ["red car", "red object", "person in red"],
        "enter": ["person entering", "car entering", "animal entering"],
        "leave": ["person leaving", "car leaving"],
    }
    results = []
    for kw, group in suggestions.items():
        if kw in text.lower():
            results.extend(group)
    if not results and text.strip():
        results = [
            f"person {text}",
            f"{text} near entrance",
            f"{text} moving",
            f"find {text}"
        ]
    return results[:6]
