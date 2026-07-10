param([string]$VideoPath = "")

$api = "http://localhost:8000/api"
$pass = $true

function Check { param($Name, $Result, $Expected)
    if ($Result -match $Expected) { Write-Host "  PASS: $Name" -ForegroundColor Green }
    else { Write-Host "  FAIL: $Name -> $Result" -ForegroundColor Red; $script:pass = $false }
}

Write-Host "`n=== SceneTrace AI Workflow Test ===" -ForegroundColor Cyan

# 1. Health
Write-Host "[1] Health check" -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/health" -TimeoutSec 10
    Check "GET /api/health" ($r.status) "ok"
} catch { Write-Host "  FAIL: Health check - is the backend running?" -ForegroundColor Red; exit 1 }

# 2. Video
if (-not $VideoPath -or -not (Test-Path $VideoPath)) {
    Write-Host "[2] No video specified. Create one: python -c `"import cv2,numpy as np; w,h,f,d=320,240,30,5; o=cv2.VideoWriter('t.mp4',cv2.VideoWriter_fourcc(*'mp4v'),f,(w,h)); [o.write((lambda t:cv2.putText(cv2.circle(np.zeros((h,w,3),np.uint8),(int(100+50*np.sin(2*np.pi*t/f)),120),20,(0,0,255),-1),f't={t/f:.1f}s',(10,30),0,0.7,(255,255,255),2))(t)) for t in range(f*d)]; o.release()"`" -ForegroundColor Yellow
    Write-Host "    Then re-run: .\test_workflow.ps1 -VideoPath t.mp4" -ForegroundColor Yellow
    exit 1
}

Write-Host "[2] Using video: $VideoPath ($((Get-Item $VideoPath).Length) bytes)" -ForegroundColor Yellow

# 3. Upload (use Python to POST multipart since PS5 lacks -Form)
Write-Host "[3] Uploading..." -ForegroundColor Yellow
try {
    $py = @"
import httpx, sys
with open('$($VideoPath.Replace('\','\\'))', 'rb') as f:
    r = httpx.post('$api/videos/upload', files={'file': ('video.mp4', f, 'video/mp4')}, timeout=30)
    print(r.json()['video_id'])
"@
    $vid = python -c $py 2>$null
    if (-not $vid) { throw "No video_id returned" }
    $vid = $vid.Trim()
    Check "Upload -> video_id" $vid "\w{8}"
    Write-Host "  video_id: $vid" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Upload - $_" -ForegroundColor Red; exit 1 }

# 4. Index (async - poll progress until done)
Write-Host "[4] Indexing (may take time for long videos)..." -ForegroundColor Yellow
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/index" -Method Post -TimeoutSec 30
    if ($r.status -ne "indexing_started") { throw "Expected indexing_started, got $($r.status)" }
    Write-Host "  Index started, polling progress..." -ForegroundColor Gray
    $done = $false
    do {
        Start-Sleep -Milliseconds 800
        $p = Invoke-RestMethod -Uri "$api/videos/$vid/index-progress" -TimeoutSec 10
        if ($p.stage -eq "done") { $done = $true }
        elseif ($p.stage -eq "error") { throw "Index error: $($p.message)" }
        else { Write-Host "    $($p.message) - $($p.percent)%" -ForegroundColor Gray }
    } while (-not $done)
    $sw.Stop()
    Write-Host "  $($p.keyframes) keyframes indexed in $($sw.Elapsed.TotalSeconds.ToString('0.0'))s" -ForegroundColor Gray
    Check "Index -> keyframes" ($p.keyframes) "\d+"
} catch { Write-Host "  FAIL: Index - $_" -ForegroundColor Red; exit 1 }

# 5. Status
Write-Host "[5] Status..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/videos/$vid/status" -TimeoutSec 10
    Check "Status -> ready" ($r.status) "ready"
} catch { Write-Host "  FAIL: Status - $_" -ForegroundColor Red; $pass = $false }

# 6. Search
Write-Host "[6] Searching..." -ForegroundColor Yellow
try {
    $body = '{ "query": "a red circle moving", "top_k": 3 }'
    $r = Invoke-RestMethod -Uri "$api/search" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Write-Host "  Status: $($r.status), Segments: $($r.segments.Count)" -ForegroundColor Gray
    Check "Search -> status" ($r.status) "high|medium|low"
    if ($r.segments.Count -gt 0) {
        Write-Host "  Top score: $(($r.segments[0].avg_score * 100).ToString('0.1'))%" -ForegroundColor Gray
    }
} catch { Write-Host "  FAIL: Search - $_" -ForegroundColor Red; $pass = $false }

# 7. Report
Write-Host "[7] Report..." -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$api/reports/$vid" -TimeoutSec 10
    Check "Report" ($r.frame_reduction_pct) "\d+"
    Write-Host "  Frame reduction: $($r.frame_reduction_pct)%" -ForegroundColor Gray
} catch { Write-Host "  FAIL: Report - $_" -ForegroundColor Red; $pass = $false }

Write-Host "`n==============================" -ForegroundColor Cyan
if ($pass) { Write-Host "RESULT: ALL TESTS PASSED" -ForegroundColor Green; exit 0 }
else { Write-Host "RESULT: SOME TESTS FAILED" -ForegroundColor Red; exit 1 }
