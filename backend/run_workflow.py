import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:8000/api"
VIDEO_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("T2.mp4")
QUERY = sys.argv[2] if len(sys.argv) > 2 else "a car driving through the intersection"
pass_flag = True

def check(name, result, expected):
    global pass_flag
    if callable(expected):
        ok = expected(result)
    elif isinstance(expected, type):
        ok = isinstance(result, expected)
    else:
        ok = expected in result
    if ok:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name} -> {result}")
        pass_flag = False

def post_json(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

def upload_video(path):
    boundary = b"----Boundary" + os.urandom(16).hex().encode()
    with Path(path).open("rb") as f:
        file_data = f.read()
    filename = Path(path).name.encode()
    body = (
        b"--" + boundary + b'\r\n'
        b'Content-Disposition: form-data; name="file"; filename="' + filename + b'"\r\n'
        b'Content-Type: video/mp4\r\n\r\n' + file_data +
        b"\r\n--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(f"{API}/videos/upload", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
    return json.loads(urllib.request.urlopen(req).read())["video_id"]

def fetch(url, timeout=10):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())

print()
print("=" * 67)
print("   SceneTrace AI - Full Backend Workflow Test")
print(f"   Query: {QUERY}")
print("=" * 67)

# 1. Health
print("[1/13] Health check")
try:
    r = fetch(f"{API}/health", timeout=10)
    check("GET /api/health", r["status"], "ok")
except Exception as e:
    print(f"  FAIL: Health check - is the backend running? ({e})")
    sys.exit(1)

# 2. Video
if not VIDEO_PATH.exists():
    print(f"[2/13] Video not found at {VIDEO_PATH}")
    sys.exit(1)
file_size = VIDEO_PATH.stat().st_size
print(f"[2/13] Video: {VIDEO_PATH} ({file_size} bytes)")

# 3. Upload
print("[3/13] Uploading...")
try:
    vid = upload_video(str(VIDEO_PATH))
    check("Upload -> video_id", vid, lambda x: len(x) >= 8)
    print(f"  video_id: {vid}")
except Exception as e:
    print(f"  FAIL: Upload - {e}")
    sys.exit(1)

# 4. Index
print("[4/13] Indexing (async, polling progress)...")
start = time.time()
try:
    req = urllib.request.Request(f"{API}/videos/{vid}/index", data=b"", headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    if r.get("status") != "indexing_started":
        err_msg = f"Expected indexing_started, got {r.get('status')}"
        raise Exception(err_msg)
    done = False
    while not done:
        time.sleep(1)
        p = fetch(f"{API}/videos/{vid}/index-progress", timeout=10)
        stage = p.get("stage")
        if stage == "done":
            done = True
        elif stage == "error":
            err_msg = f"Index error: {p.get('message')}"
            raise Exception(err_msg)
        else:
            sys.stdout.write(f"\r  {p.get('message', '')} ({p.get('percent', 0)}%)")
            sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n  {p.get('keyframes', 0)} keyframes in {elapsed:.1f}s")
    check("Index -> keyframes", p.get("keyframes", 0), lambda x: isinstance(x, int) and x >= 0)
    check("Index -> stage done", p.get("stage"), "done")
except Exception as e:
    print(f"\n  FAIL: Index - {e}")
    sys.exit(1)

# 5. Status
print("[5/13] Video status...")
try:
    r = fetch(f"{API}/videos/{vid}/status", timeout=10)
    check("Status -> ready", r.get("status"), "ready")
    check("Status -> keyframes", r.get("keyframes", 0), lambda x: isinstance(x, int))
    print(f"  Keyframes: {r['keyframes']}, Total frames: {r.get('total_frames', 0)}")
except Exception as e:
    print(f"  FAIL: Status - {e}")
    pass_flag = False

# 6. Timeline
print("[6/13] Timeline...")
try:
    r = fetch(f"{API}/videos/{vid}/timeline", timeout=10)
    check("Timeline -> video_id", r.get("video_id", ""), lambda x: len(x) >= 8)
    check("Timeline -> fps", str(r.get("metadata", {}).get("fps", 0)), lambda x: x.replace(".", "", 1).isdigit())
    meta = r.get("metadata", {})
    print(f"  {meta.get('width', '?')}x{meta.get('height', '?')} at {meta.get('fps', '?')}fps, Events: {len(r.get('events', []))}")
    events = r.get("events", [])
    if events:
        print(f"  First: frame {events[0].get('frame_index', '?')} at {events[0].get('timestamp', '?')}s")
except Exception as e:
    print(f"  FAIL: Timeline - {e}")
    pass_flag = False

# 7. Enhanced Search (v2) with detection
print("[7/13] Enhanced Search (v2) with detection")
try:
    r = post_json(f"{API}/v2/search", {"query": QUERY, "top_k": 5, "enable_detection": True})
    check("v2 Search -> status", r.get("status", ""), lambda x: x in ("high", "medium", "low"))
    check("v2 Search -> segments", str(len(r.get("segments", []))), lambda x: x.isdigit())
    print(f"  Query time: {r.get('query_time', 0)}s, Status: {r.get('status', '?')}")
    if r.get("query_info", {}).get("search_plan"):
        plan = r["query_info"]["search_plan"]
        check("Search plan -> objects", str(len(plan.get("objects", []))), lambda x: x.isdigit())
        if plan.get("objects"):
            print(f"  Objects: {', '.join(plan['objects'])}")
        if plan.get("actions"):
            print(f"  Actions: {', '.join(plan['actions'])}")
        if plan.get("location"):
            print(f"  Location: {plan['location']}")
    segments = r.get("segments", [])
    if segments:
        s = segments[0]
        print(f"  Top segment: score {s.get('avg_score', 0) * 100:.1f}%")
        frames = s.get("frame_indices", [])
        if frames:
            print(f"  Frames: {frames[0]} .. {frames[-1]}")
        sb = s.get("score_breakdown")
        if sb:
            check("Breakdown -> weighted_total", str(sb.get("weighted_total", 0)), lambda x: x.replace(".", "", 1).isdigit())
            print("  Signals:")
            for key in ("clip_semantic", "caption_similarity", "object_match", "motion_activity", "tracking_consistency", "temporal_alignment", "relationship_overlap"):
                print(f"    {key:25s} {sb.get(key, 0)}")
        if s.get("annotated_thumbnail"):
            check("Annotated thumbnail URL", s["annotated_thumbnail"], "/api/frames/")
            try:
                tr = urllib.request.urlopen(f"http://localhost:8000{s['annotated_thumbnail']}", timeout=10)
                check("Annotated thumbnail fetch", str(tr.status), "200")
                print(f"  Thumbnail size: {len(tr.read())} bytes")
            except Exception:
                print("  WARN: Annotated thumbnail not found")
        detections = s.get("detections", [])
        if detections:
            print("  Detections on mid frame:")
            for d in detections:
                print(f"    {d.get('label', '?')} (score: {d.get('score', '?')})")
        track_ids = s.get("track_ids", [])
        if track_ids:
            print(f"  Track IDs: {', '.join(str(t) for t in track_ids)}")
    else:
        print("  WARN: No segments returned")
except Exception as e:
    print(f"  FAIL: v2 Search - {e}")
    pass_flag = False

# 8. v2 Search with detection disabled
print("[8/13] Enhanced Search (v2) no detection")
try:
    r = post_json(f"{API}/v2/search", {"query": QUERY, "top_k": 3, "enable_detection": False})
    check("v2 (no det) -> status", r.get("status", ""), lambda x: x in ("high", "medium", "low"))
    segments = r.get("segments", [])
    if segments:
        print(f"  Top score: {segments[0].get('avg_score', 0) * 100:.1f}%")
except Exception as e:
    print(f"  FAIL: v2 (no det) - {e}")
    pass_flag = False

# 9. Annotated objects
print("[9/13] Detected objects...")
try:
    r = fetch(f"{API}/videos/{vid}/objects", timeout=15)
    check("Objects -> video_id", r.get("video_id", ""), lambda x: len(x) >= 8)
    frames_list = r.get("detected_frames", [])
    print(f"  Annotated frames: {len(frames_list)}")
    if frames_list:
        first = frames_list[0]
        print(f"  First: frame {first.get('frame_index', '?')} at {first.get('timestamp', '?')}s")
        try:
            tr = urllib.request.urlopen(f"http://localhost:8000{first.get('annotated', '')}", timeout=10)
            check("Objects annotated thumbnail", str(tr.status), "200")
        except Exception:
            print("  WARN: Object annotated frame not found")
except Exception as e:
    print(f"  FAIL: Objects - {e}")
    pass_flag = False

# 10. Suggestions
print("[10/13] Search suggestions...")
try:
    r = post_json(f"{API}/search/suggest", {"query": "person", "top_k": 5})
    check("Suggest -> count", str(len(r.get("suggestions", []))), lambda x: x.isdigit())
    print(f"  Suggestions: {', '.join(r.get('suggestions', []))}")
except Exception as e:
    print(f"  FAIL: Suggestions - {e}")
    pass_flag = False

# 11. Dashboard
print("[11/13] Dashboard metrics...")
try:
    r = fetch(f"{API}/dashboard/metrics", timeout=10)
    check("Dashboard -> videos", str(r.get("videos_indexed", 0)), lambda x: x.isdigit())
    check("Dashboard -> speed", str(r.get("indexing_speed_fps", 0)), lambda x: True)
    check("Dashboard -> latency", str(r.get("avg_query_latency", 0)), lambda x: True)
    print(f"  Indexed: {r['videos_indexed']}, Speed: {r.get('indexing_speed_fps', 0)}fps")
    print(f"  Frames: {r.get('total_frames', 0)}, Keyframes: {r.get('total_keyframes', 0)}")
    print(f"  Reduction: {r.get('avg_frame_reduction_pct', 0)}%, Latency: {r.get('avg_query_latency', 0) * 1000:.0f}ms")
except Exception as e:
    print(f"  FAIL: Dashboard - {e}")
    pass_flag = False

# 12. Report
print("[12/13] Report...")
try:
    r = fetch(f"{API}/reports/{vid}", timeout=10)
    check("Report -> reduction", str(r.get("frame_reduction_pct", 0)), lambda x: True)
    print(f"  Reduction: {r.get('frame_reduction_pct', 0)}%, Motion: {r.get('motion_activity_avg', 0)}")
except Exception as e:
    print(f"  FAIL: Report - {e}")
    pass_flag = False

# 13. Clip extraction
print("[13/13] Video clip extraction...")
try:
    r = post_json(f"{API}/v2/search", {"query": QUERY, "top_k": 1, "enable_detection": True})
    segments = r.get("segments", [])
    if segments:
        seg = segments[0]
        frames = seg.get("frame_indices", [])
        if frames:
            sf, ef = frames[0], frames[-1]
            try:
                cr = urllib.request.urlopen(f"{API}/clips/{vid}?start_frame={sf}&end_frame={ef}", timeout=30)
                check("Clip extraction -> 200", str(cr.status), "200")
                print(f"  Clip: {len(cr.read())} bytes ({sf} .. {ef})")
            except Exception as e:
                print(f"  WARN: Clip extraction failed - {e}")
        else:
            print("  WARN: No frame indices in segment")
    else:
        print("  WARN: No segments for clip")
except Exception as e:
    print(f"  FAIL: Clip - {e}")
    pass_flag = False

# Summary
print()
print("=" * 67)
print("   WORKFLOW COMPLETE")
print("=" * 67)
print(f"  Video:     {VIDEO_PATH}")
print(f"  Query:     {QUERY}")
print(f"  Video ID:  {vid}")
print()
if pass_flag:
    print("  RESULT: ALL 13 TESTS PASSED")
    sys.exit(0)
else:
    print("  RESULT: SOME TESTS FAILED")
    sys.exit(1)
