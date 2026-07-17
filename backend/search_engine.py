"""Enhanced search: CLIP + object detection + weighted scoring + explanations."""
import time
import cv2
import numpy as np
import logging
from pathlib import Path
from cachetools import TTLCache
from pipeline import STORAGE, search_embeddings, frames_to_segments, embed_text
from config import benchmark
from config import settings

logger = logging.getLogger("scenetrace.search")

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

def _search_objects_faiss(query: str, vid: str, idx,
                          top_k: int = 15) -> dict[int, list[dict]]:
    """Query the object-level FAISS index and return {frame_idx: [match_obj, ...]}."""
    obj_faiss = idx.get_object_faiss_index()
    obj_embs = idx.load_object_embeddings()
    if obj_faiss is None or obj_embs is None or len(obj_embs) == 0:
        return {}
    if len(idx.object_metadata) != len(obj_embs):
        return {}
    q_emb = embed_text([query])
    n = len(obj_embs)
    scores, indices = obj_faiss.search(q_emb, min(top_k * 3, n))
    frame_map: dict[int, list[dict]] = {}
    for obj_idx, score in zip(indices[0], scores[0]):
        meta = idx.object_metadata[obj_idx]
        fi = meta["frame_idx"]
        entry = {
            "bbox": meta["bbox"],
            "label": meta["label"],
            "score": meta["score"],
            "clip_similarity": round(float(score), 3),
        }
        if fi not in frame_map:
            frame_map[fi] = []
        frame_map[fi].append(entry)
    return frame_map


def search(query: str, indexes: dict, top_k: int = 5, enable_detection: bool = True) -> dict:
    t0 = time.time()
    if not indexes:
        return {"segments": [], "status": "no_indexes", "query_info": {"raw": query}}

    candidates = []
    object_frame_maps: dict[str, dict[int, list[dict]]] = {}
    for vid, idx in indexes.items():
        embs = idx.load_embeddings()
        if embs is None or len(embs) == 0:
            continue
        faiss_idx = idx.get_faiss_index()
        indices, scores = search_embeddings(query, faiss_idx, embs, top_k * 3)
        segs = frames_to_segments(indices, scores)
        for s in segs:
            orig = s["frame_indices"]
            s["video_id"] = vid
            s["frame_indices"] = [idx.frame_indices[i] for i in orig]
            s["timestamps"] = [idx.timestamps[i] for i in orig]
            s["semantic_score"] = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        segs.sort(key=lambda s: s["semantic_score"], reverse=True)
        candidates.extend(segs[:top_k * 2])

        if enable_detection:
            obj_map = _search_objects_faiss(query, vid, idx, top_k)
            if obj_map:
                object_frame_maps[vid] = obj_map

    candidates.sort(key=lambda s: s["semantic_score"], reverse=True)
    segments = candidates[:top_k]

    if enable_detection:
        for seg in segments:
            mid = seg["frame_indices"][len(seg["frame_indices"]) // 2]
            vid = seg["video_id"]
            use_indexed = vid in object_frame_maps and mid in object_frame_maps[vid]
            if use_indexed:
                objs = object_frame_maps[vid][mid]
                seg["detections"] = objs
                sims = [o["clip_similarity"] for o in objs]
                seg["object_score"] = round(sum(sims) / len(sims), 4) if sims else 0
            else:
                try:
                    dets = _get_detections_for_segment(seg, query)
                    seg["detections"] = dets
                    obj_scores = [d["score"] * 0.5 for d in dets]
                    seg["object_score"] = round(sum(obj_scores) / max(len(obj_scores), 1), 4) if obj_scores else 0
                except Exception:
                    logger.exception("Detection error on %s", seg.get("video_id"))
                    seg["detections"] = []
                    seg["object_score"] = 0
    else:
        for seg in segments:
            seg["detections"] = []
            seg["object_score"] = 0

    for seg in segments:
        s = seg["semantic_score"]
        o = seg["object_score"]
        w = round(settings.SEMANTIC_WEIGHT * s + settings.OBJECT_WEIGHT * o, 4)
        seg["weighted_score"] = w
        seg["score_breakdown"] = {
            "semantic_similarity": round(s, 3),
            "object_match": round(o, 3),
            "tracking_consistency": 0,
            "temporal_match": 0,
            "motion_activity": 0,
            "weighted_total": w
        }
        seg["avg_score"] = w

    segments.sort(key=lambda s: s["weighted_score"], reverse=True)
    top_score = segments[0]["weighted_score"] if segments else 0
    status = "high" if top_score > settings.SEARCH_HIGH_THRESHOLD else ("medium" if top_score > settings.SEARCH_MEDIUM_THRESHOLD else "low")

    elapsed = time.time() - t0
    benchmark.record_query(elapsed)

    return {
        "segments": segments,
        "status": status,
        "query_info": {"raw": query, "semantic_query": query},
        "query_time": round(elapsed, 2)
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
