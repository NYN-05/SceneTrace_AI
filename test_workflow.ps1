param([string]$VideoPath = "")

$api = "http://localhost:8000/api"
$pass = $true
$tmp = [System.IO.Path]::GetTempPath()

function Check { param($Name, $Result, $Expected)
    if ($Result -match $Expected) { Write-Host "  PASS: $Name" -ForegroundColor Green }
    else { Write-Host "  FAIL: $Name -> $Result" -ForegroundColor Red; $script:pass = $false }
}

function Write-Python { param($Name, $Code)
    $path = Join-Path $tmp "$Name.py"
    $Code | Set-Content -Path $path -Force
    return $path
}

Write-Host "`n===================================================================" -ForegroundColor Cyan
Write-Host "   SceneTrace AI - Full Workflow + Enhanced Features Test" -ForegroundColor Cyan
Write-Host "===================================================================" -ForegroundColor Cyan

# 1. Health
Write-Host "[1/12] Health check" -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/health" -TimeoutSec 10
    Check "GET /api/health" ($r.status) "ok"
} catch { Write-Host "  FAIL: Health check - is the backend running?" -ForegroundColor Red; exit 1 }

# 2. Video
if (-not $VideoPath -or -not (Test-Path $VideoPath)) {
    Write-Host "[2/12] No video found. Generate one: .\test_workflow.ps1 -VideoPath test.mp4" -ForegroundColor Yellow
    exit 1
}
$fileSize = (Get-Item $VideoPath).Length
Write-Host "[2/12] Video: $VideoPath ($fileSize bytes)" -ForegroundColor Yellow

# 3. Upload
Write-Host "[3/12] Uploading..." -ForegroundColor Yellow
$uploadUrl = "$api/videos/upload"
$uploadPy = Write-Python "upload" @"
import httpx, sys
path = sys.argv[1]
url = sys.argv[2]
with open(path, 'rb') as f:
    r = httpx.post(url, files={'file': ('video.mp4', f, 'video/mp4')}, timeout=30)
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
Write-Host "[4/12] Indexing (async, polling progress)..." -ForegroundColor Yellow
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
Write-Host "[5/12] Status..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/status" -TimeoutSec 10
    Check "Status -> ready" ($r.status) "ready"
    Check "Status -> keyframes" ($r.keyframes) "\d+"
    Write-Host "  Keyframes: $($r.keyframes), Total: $($r.total_frames)" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Status - $_" -ForegroundColor Red; $pass = $false }

# 6. Original search (v1, backward compat)
Write-Host "[6/12] Original Search (v1)" -ForegroundColor Yellow
try {
    $body = '{"query":"a red circle moving","top_k":3}'
    $r = Invoke-RestMethod -Uri "$api/search" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Check "v1 Search -> status" ($r.status) "high|medium|low"
    Write-Host "  Segments: $($r.segments.Count), Top: $(($r.segments[0].avg_score * 100).ToString('0.1'))%" -ForegroundColor Gray
} catch { Write-Host "  FAIL: v1 Search - $_" -ForegroundColor Red; $pass = $false }

# 7. Report
Write-Host "[7/12] Report..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/reports/$vid" -TimeoutSec 10
    Check "Report -> reduction" ($r.frame_reduction_pct.ToString()) "\d+"
    Write-Host "  Reduction: $($r.frame_reduction_pct)%, Motion: $($r.motion_activity_avg)" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Report - $_" -ForegroundColor Red; $pass = $false }

# 8. Enhanced search v2
Write-Host "[8/12] Enhanced Search (v2)" -ForegroundColor Yellow
try {
    $body = '{"query":"red circle","top_k":5,"enable_detection":false}'
    $r = Invoke-RestMethod -Uri "$api/v2/search" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Check "v2 Search -> status" ($r.status) "high|medium|low"
    Check "v2 Search -> segments" ($r.segments.Count.ToString()) "\d+"
    if ($r.segments.Count -gt 0) {
        $s = $r.segments[0]
        Write-Host "  Top: $(($s.avg_score * 100).ToString('0.1'))%, Latency: $($r.query_time)s" -ForegroundColor Gray
        if ($s.score_breakdown) {
            Check "v2 -> score_breakdown" ($s.score_breakdown.semantic_similarity.ToString()) "\d"
            Write-Host "  Semantic: $($s.score_breakdown.semantic_similarity), Object: $($s.score_breakdown.object_match)" -ForegroundColor Gray
        }
    }
} catch { Write-Host "  FAIL: v2 Search - $_" -ForegroundColor Red; $pass = $false }

# 9. Suggestions
Write-Host "[9/12] Search Suggestions..." -ForegroundColor Yellow
try {
    $body = '{"query":"person","top_k":3}'
    $r = Invoke-RestMethod -Uri "$api/search/suggest" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 10
    Check "Suggest -> count" ($r.suggestions.Count.ToString()) "\d+"
    Write-Host "  Suggestions: $($r.suggestions -join ', ')" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Suggestions - $_" -ForegroundColor Red; $pass = $false }

# 10. Dashboard metrics
Write-Host "[10/12] Dashboard Metrics..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/dashboard/metrics" -TimeoutSec 10
    Check "Dashboard -> videos" ($r.videos_indexed.ToString()) "\d+"
    Check "Dashboard -> speed" ($r.indexing_speed_fps.ToString()) "\d"
    Check "Dashboard -> latency" ($r.avg_query_latency.ToString()) "\d"
    Write-Host "  Indexed: $($r.videos_indexed), Speed: $($r.indexing_speed_fps)fps" -ForegroundColor Gray
    Write-Host "  Frames: $($r.total_frames), Keyframes: $($r.total_keyframes), Reduction: $($r.avg_frame_reduction_pct)%" -ForegroundColor Gray
    Write-Host "  Avg latency: $(($r.avg_query_latency * 1000).ToString('0'))ms, Queries: $($r.total_queries)" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Dashboard - $_" -ForegroundColor Red; $pass = $false }

# 11. Timeline
Write-Host "[11/12] Timeline..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/timeline" -TimeoutSec 10
    Check "Timeline -> video_id" ($r.video_id) "\w+"
    Check "Timeline -> fps" ($r.metadata.fps.ToString()) "\d"
    Write-Host "  $($r.metadata.width)x$($r.metadata.height) @ $($r.metadata.fps)fps, Events: $($r.events.Count)" -ForegroundColor Gray
    if ($r.events.Count -gt 0) {
        Write-Host "  First: frame $($r.events[0].frame_index) @ $($r.events[0].timestamp)s" -ForegroundColor Gray
    }
} catch { Write-Host "  FAIL: Timeline - $_" -ForegroundColor Red; $pass = $false }

# 12. Objects
Write-Host "[12/12] Detected Objects..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/objects" -TimeoutSec 15
    Check "Objects -> video_id" ($r.video_id) "\w+"
    Write-Host "  Annotated frames: $($r.detected_frames.Count)" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Objects - $_" -ForegroundColor Red; $pass = $false }

# Summary
Write-Host "`n===================================================================" -ForegroundColor Cyan
if ($pass) {
    Write-Host "RESULT: ALL 12 TESTS PASSED" -ForegroundColor Green
    Write-Host "  Pipeline: health - upload - index - status - search(v1) - report" -ForegroundColor Gray
    Write-Host "  Enhanced: v2search - suggestions - dashboard - timeline - objects" -ForegroundColor Gray
    exit 0
} else {
    Write-Host "RESULT: SOME TESTS FAILED" -ForegroundColor Red
    exit 1
}
