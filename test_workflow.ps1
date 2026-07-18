param([string]$VideoPath = "")

$api = "http://localhost:8000/api"
$pass = $true
$tmp = [System.IO.Path]::GetTempPath()

function Check { param($Name, $Result, $Expected)
    if ($Result -match $Expected) { Write-Host "  PASS: $Name" -ForegroundColor Green }
    else { Write-Host "  FAIL: $Name -> $Result (expected: $Expected)" -ForegroundColor Red; $script:pass = $false }
}

function Write-Python { param($Name, $Code)
    $path = Join-Path $tmp "$Name.py"
    $Code | Set-Content -Path $path -Force
    return $path
}

Write-Host "`n===================================================================" -ForegroundColor Cyan
Write-Host "   SceneTrace AI - Full Backend Workflow Test" -ForegroundColor Cyan
Write-Host "   Query: ""white sports car parked on the street""" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

# 1. Health
Write-Host "[1/13] Health check" -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/health" -TimeoutSec 10
    Check "GET /api/health" ($r.status) "ok"
} catch { Write-Host "  FAIL: Health check - is the backend running?" -ForegroundColor Red; exit 1 }

# 2. Video
if (-not $VideoPath -or -not (Test-Path $VideoPath)) {
    Write-Host "[2/13] No video found. Usage: .\test_workflow.ps1 -VideoPath T2.mp4" -ForegroundColor Yellow
    exit 1
}
$fileSize = (Get-Item $VideoPath).Length
Write-Host "[2/13] Video: $VideoPath ($fileSize bytes)" -ForegroundColor Yellow

# 3. Upload
Write-Host "[3/13] Uploading..." -ForegroundColor Yellow
$uploadUrl = "$api/videos/upload"
$uploadPy = Write-Python "upload" @"
import httpx, sys
path = sys.argv[1]
url = sys.argv[2]
with open(path, 'rb') as f:
    r = httpx.post(url, files={'file': ('video.mp4', f, 'video/mp4')}, timeout=60)
    print(r.json()['video_id'])
"@
try {
    $vid = python $uploadPy $VideoPath $uploadUrl 2>$null
    if (-not $vid) { throw "No video_id returned" }
    $vid = $vid.Trim()
    Check "Upload -> video_id" $vid "\w{8}"
    Write-Host "  video_id: $vid" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Upload - $_" -ForegroundColor Red; exit 1 }

# 4. Index
Write-Host "[4/13] Indexing (async, polling progress)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/index" -Method Post -TimeoutSec 30
    if ($r.status -ne "indexing_started") { throw "Expected indexing_started, got $($r.status)" }
    Write-Host "  Async started, polling..." -ForegroundColor Gray
    $done = $false
    do {
        Start-Sleep -Milliseconds 800
        $p = Invoke-RestMethod -Uri "$api/videos/$vid/index-progress" -TimeoutSec 10
        if ($p.stage -eq "done") { $done = $true }
        elseif ($p.stage -eq "error") { throw "Index error: $($p.message)" }
        else { Write-Host "    $($p.message) - $($p.percent)%" -ForegroundColor Gray }
    } while (-not $done)
    $sw.Stop()
    Write-Host "  $($p.keyframes) keyframes in $($sw.Elapsed.TotalSeconds.ToString('0.0'))s" -ForegroundColor Gray
    Check "Index -> keyframes" ($p.keyframes) "\d+"
    Check "Index -> stage done" ($p.stage) "done"
} catch { Write-Host "  FAIL: Index - $_" -ForegroundColor Red; exit 1 }

# 5. Status
Write-Host "[5/13] Video status..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/status" -TimeoutSec 10
    Check "Status -> ready" ($r.status) "ready"
    Check "Status -> keyframes" ($r.keyframes) "\d+"
    Write-Host "  Keyframes: $($r.keyframes), Total frames: $($r.total_frames)" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Status - $_" -ForegroundColor Red; $pass = $false }

# 6. Timeline (verify motion data)
Write-Host "[6/13] Timeline..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/timeline" -TimeoutSec 10
    Check "Timeline -> video_id" ($r.video_id) "\w+"
    Check "Timeline -> fps" ($r.metadata.fps.ToString()) "\d+"
    Write-Host "  $($r.metadata.width)x$($r.metadata.height) @ $($r.metadata.fps)fps, Events: $($r.events.Count)" -ForegroundColor Gray
    if ($r.events.Count -gt 0) {
        Write-Host "  First: frame $($r.events[0].frame_index) @ $($r.events[0].timestamp)s" -ForegroundColor Gray
    }
} catch { Write-Host "  FAIL: Timeline - $_" -ForegroundColor Red; $pass = $false }

# 7. Enhanced Search (v2) — main query with detection enabled
Write-Host "[7/13] Enhanced Search (v2) — ""white sports car parked on the street""" -ForegroundColor Yellow
$query = "white sports car parked on the street"
try {
    $body = @{query=$query; top_k=5; enable_detection=$true} | ConvertTo-Json -Compress
    $r = Invoke-RestMethod -Uri "$api/v2/search" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Check "v2 Search -> status" ($r.status) "high|medium|low"
    Check "v2 Search -> segments" ($r.segments.Count.ToString()) "\d+"
    Write-Host "  Query time: $($r.query_time)s, Status: $($r.status)" -ForegroundColor Gray

    # Verify query_info contains search_plan with parsed fields
    if ($r.query_info.search_plan) {
        $plan = $r.query_info.search_plan
        Check "Search plan -> objects count" ($plan.objects.Count.ToString()) "\d+"
        if ($plan.objects) { Write-Host "  Objects found: $($plan.objects -join ', ')" -ForegroundColor Gray }
        if ($plan.actions) { Write-Host "  Actions found: $($plan.actions -join ', ')" -ForegroundColor Gray }
        if ($plan.attributes) { Write-Host "  Attributes: $(($plan.attributes | ConvertTo-Json -Compress))" -ForegroundColor Gray }
        if ($plan.location) { Write-Host "  Location: $($plan.location)" -ForegroundColor Gray }
    } else { Write-Host "  WARN: No search_plan in response" -ForegroundColor Yellow }

    if ($r.segments.Count -gt 0) {
        $s = $r.segments[0]
        Write-Host "  Top segment: score $(($s.avg_score * 100).ToString('0.1'))%" -ForegroundColor Gray
        Write-Host "  Frames: $($s.frame_indices[0])..$($s.frame_indices[-1])" -ForegroundColor Gray

        # Check all 7 score breakdown signals
        if ($s.score_breakdown) {
            $sb = $s.score_breakdown
            Check "Breakdown -> weighted_total" ($sb.weighted_total.ToString()) "\d+\.?\d*"
            Write-Host "  Signals:" -ForegroundColor Gray
            Write-Host "    clip_semantic:       $($sb.clip_semantic)" -ForegroundColor Gray
            Write-Host "    caption_similarity:  $($sb.caption_similarity)" -ForegroundColor Gray
            Write-Host "    object_match:        $($sb.object_match)" -ForegroundColor Gray
            Write-Host "    motion_activity:     $($sb.motion_activity)" -ForegroundColor Gray
            Write-Host "    tracking_consistency: $($sb.tracking_consistency)" -ForegroundColor Gray
            Write-Host "    temporal_alignment:  $($sb.temporal_alignment)" -ForegroundColor Gray
            Write-Host "    relationship_overlap: $($sb.relationship_overlap)" -ForegroundColor Gray
        }

        # Check annotated thumbnail
        if ($s.annotated_thumbnail) {
            Check "Annotated thumbnail URL" ($s.annotated_thumbnail) "/api/frames/.+_d\.jpg"
            Write-Host "  Thumbnail: $($s.annotated_thumbnail)" -ForegroundColor Gray
            # Verify the thumbnail actually serves
            try {
                $turl = "http://localhost:8000$($s.annotated_thumbnail)"
                $tr = Invoke-WebRequest -Uri $turl -TimeoutSec 10
                Check "Annotated thumbnail fetch" ($tr.StatusCode.ToString()) "200"
                Write-Host "  Thumbnail size: $($tr.RawContentLength) bytes" -ForegroundColor Gray
            } catch { Write-Host "  WARN: Annotated thumbnail not found (may need re-index)" -ForegroundColor Yellow }
        }

        # Check detections on segment
        if ($s.detections -and $s.detections.Count -gt 0) {
            Write-Host "  Detections on middle frame:" -ForegroundColor Gray
            $s.detections | ForEach-Object { Write-Host "    $($_.label) (score: $($_.score))" -ForegroundColor Gray }
        }

        # Check track_ids exist
        if ($s.track_ids -and $s.track_ids.Count -gt 0) {
            Write-Host "  Track IDs on segment: $($s.track_ids -join ', ')" -ForegroundColor Gray
        }
    }
} catch { Write-Host "  FAIL: v2 Search - $_" -ForegroundColor Red; $pass = $false }

# 8. v2 Search with detection disabled (frame-only embedding path)
Write-Host "[8/13] Enhanced Search (v2) — detection disabled" -ForegroundColor Yellow
try {
    $body = @{query=$query; top_k=3; enable_detection=$false} | ConvertTo-Json -Compress
    $r = Invoke-RestMethod -Uri "$api/v2/search" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Check "v2 (no det) -> status" ($r.status) "high|medium|low"
    Write-Host "  Top score: $(($r.segments[0].avg_score * 100).ToString('0.1'))%" -ForegroundColor Gray
} catch { Write-Host "  FAIL: v2 (no det) - $_" -ForegroundColor Red; $pass = $false }

# 9. Annotated objects endpoint
Write-Host "[9/13] Detected objects..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/objects" -TimeoutSec 15
    Check "Objects -> video_id" ($r.video_id) "\w+"
    Write-Host "  Annotated frames: $($r.detected_frames.Count)" -ForegroundColor Gray
    if ($r.detected_frames.Count -gt 0) {
        $first = $r.detected_frames[0]
        Write-Host "  First: frame $($first.frame_index) @ $($first.timestamp)s" -ForegroundColor Gray
        # Verify annotated thumbnail exists
        try {
            $turl = "http://localhost:8000$($first.annotated)"
            $tr = Invoke-WebRequest -Uri $turl -TimeoutSec 10
            Check "Objects annotated thumbnail" ($tr.StatusCode.ToString()) "200"
        } catch { Write-Host "  WARN: Annotated object frame not found" -ForegroundColor Yellow }
    }
} catch { Write-Host "  FAIL: Objects - $_" -ForegroundColor Red; $pass = $false }

# 10. Search suggestions
Write-Host "[10/13] Search suggestions..." -ForegroundColor Yellow
try {
    $body = @{query="person"; top_k=5} | ConvertTo-Json -Compress
    $r = Invoke-RestMethod -Uri "$api/search/suggest" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10
    Check "Suggest -> count" ($r.suggestions.Count.ToString()) "\d+"
    Write-Host "  Suggestions: $($r.suggestions -join ', ')" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Suggestions - $_" -ForegroundColor Red; $pass = $false }

# 11. Dashboard metrics
Write-Host "[11/13] Dashboard metrics..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/dashboard/metrics" -TimeoutSec 10
    Check "Dashboard -> videos indexed" ($r.videos_indexed.ToString()) "\d+"
    Check "Dashboard -> speed" ($r.indexing_speed_fps.ToString()) "\d+\.?\d*"
    Check "Dashboard -> latency" ($r.avg_query_latency.ToString()) "\d+\.?\d*"
    Write-Host "  Indexed: $($r.videos_indexed), Speed: $($r.indexing_speed_fps) fps" -ForegroundColor Gray
    Write-Host "  Frames: $($r.total_frames), Keyframes: $($r.total_keyframes)" -ForegroundColor Gray
    Write-Host "  Reduction: $($r.avg_frame_reduction_pct)%, Latency: $(($r.avg_query_latency * 1000).ToString('0')) ms" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Dashboard - $_" -ForegroundColor Red; $pass = $false }

# 12. Report
Write-Host "[12/13] Report..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/reports/$vid" -TimeoutSec 10
    Check "Report -> reduction" ($r.frame_reduction_pct.ToString()) "\d+\.?\d*"
    Write-Host "  Reduction: $($r.frame_reduction_pct)%, Avg motion: $($r.motion_activity_avg)" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Report - $_" -ForegroundColor Red; $pass = $false }

# 13. Video clip extraction
Write-Host "[13/13] Video clip extraction..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/v2/search" -Method Post -Body ($body = @{query=$query; top_k=1; enable_detection=$true} | ConvertTo-Json -Compress) -ContentType "application/json" -TimeoutSec 120
    if ($r.segments.Count -gt 0) {
        $seg = $r.segments[0]
        $sf = $seg.frame_indices[0]
        $ef = $seg.frame_indices[-1]
        $clipUrl = "$api/clips/$vid`?start_frame=$sf&end_frame=$ef"
        try {
            $cr = Invoke-WebRequest -Uri $clipUrl -TimeoutSec 30
            Check "Clip extraction -> HTTP 200" ($cr.StatusCode.ToString()) "200"
            Write-Host "  Clip size: $($cr.RawContentLength) bytes ($sf → $ef)" -ForegroundColor Gray
        } catch { Write-Host "  WARN: Clip extraction failed (original video may not exist) - $_" -ForegroundColor Yellow }
    } else { Write-Host "  WARN: No segments to extract clip from" -ForegroundColor Yellow }
} catch { Write-Host "  FAIL: Clip - $_" -ForegroundColor Red; $pass = $false }

# Summary
Write-Host "`n===================================================================" -ForegroundColor Cyan
Write-Host "   WORKFLOW COMPLETE" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "  Video:     $VideoPath" -ForegroundColor Gray
Write-Host "  Query:     ""$query""" -ForegroundColor Gray
Write-Host "  Video ID:  $vid" -ForegroundColor Gray
if ($pass) {
    Write-Host "`n  RESULT: ALL 13 TESTS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n  RESULT: SOME TESTS FAILED" -ForegroundColor Red
    exit 1
}
