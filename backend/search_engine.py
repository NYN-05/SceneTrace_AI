"""Enhanced search: CLIP + object detection + weighted scoring + explanations."""
import time
import cv2
import numpy as np
from pathlib import Path
from pipeline import STORAGE, search_embeddings, frames_to_segments
from benchmark import benchmark

def _get_detections_for_segment(seg, query, cache={}):
    """Run Grounding DINO on the middle frame of a segment. Cache results."""
    vid, indices = seg["video_id"], seg["frame_indices"]
    mid = indices[len(indices) // 2]
    key = f"{vid}_{mid}"
    if key in cache:
        return cache[key]
    from detector import detect
    img_path = STORAGE / "frames" / vid / f"frame_{mid}.jpg"
    if not img_path.exists():
        cache[key] = []
        return []
    img = cv2.imread(str(img_path))
    if img is None:
        cache[key] = []
        return []
    try:
        dets = detect(img, query, threshold=0.2)
    except Exception:
        dets = []
    # Save annotated version
    if dets:
        from detector import render
        annotated = render(img, dets)
        out = STORAGE / "frames" / vid / f"frame_{mid}_d.jpg"
        cv2.imwrite(str(out), annotated)
    cache[key] = dets
    return dets

def search(query: str, indexes: dict, top_k: int = 5, enable_detection: bool = True) -> dict:
    t0 = time.time()
    if not indexes:
        return {"segments": [], "status": "no_indexes", "query_info": {"raw": query}}

    candidates = []
    for vid, idx in indexes.items():
        embs = np.array(idx.embeddings, dtype="float32")
        if len(embs) == 0:
            continue
        indices, scores = search_embeddings(query, embs, top_k * 3)
        segs = frames_to_segments(indices, scores)
        for s in segs:
            s["video_id"] = vid
            s["timestamps"] = [idx.timestamps[i] for i in s["frame_indices"]]
            s["semantic_score"] = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        segs.sort(key=lambda s: s["semantic_score"], reverse=True)
        candidates.extend(segs[:top_k * 2])

    candidates.sort(key=lambda s: s["semantic_score"], reverse=True)
    segments = candidates[:top_k]

    # Object detection on top-K
    if enable_detection:
        det_cache = {}
        for seg in segments:
            try:
                dets = _get_detections_for_segment(seg, query, det_cache)
                seg["detections"] = dets
                obj_scores = [d["score"] * 0.5 for d in dets]
                seg["object_score"] = round(sum(obj_scores) / max(len(obj_scores), 1), 4) if obj_scores else 0
            except Exception as e:
                print(f"Detection error on {seg.get('video_id')}: {e}")
                seg["detections"] = []
                seg["object_score"] = 0
    else:
        for seg in segments:
            seg["detections"] = []
            seg["object_score"] = 0

    # Weighted scoring
    for seg in segments:
        s = seg["semantic_score"]
        o = seg["object_score"]
        w = round(0.55 * s + 0.45 * o, 4)
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
    status = "high" if top_score > 0.25 else ("medium" if top_score > 0.15 else "low")

    elapsed = time.time() - t0
    benchmark.record_query(elapsed)

    return {
        "segments": segments,
        "status": status,
        "query_info": {"raw": query, "semantic_query": query},
        "query_time": round(elapsed, 2)
    }

def suggest(text: str) -> list[str]:
    """Simple query suggestions based on common patterns."""
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
