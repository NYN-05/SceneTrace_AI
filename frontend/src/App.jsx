import React, { useState, useRef, useCallback, useEffect } from "react";

const API = "";

function formatETA(sec) {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `~${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s > 0 ? `~${m}m ${s}s` : `~${m}m`;
}

function cls(...args) { return args.filter(Boolean).join(" "); }

/* Progress bar component (shared) */
function ProgressBar({ progress, startTime }) {
  if (!progress) return null;
  const pct = progress.percent || 0;
  const elapsed = startTime ? (Date.now() - startTime) / 1000 : 0;
  let eta = progress.eta_seconds;
  if (eta == null && pct > 0 && pct < 100 && elapsed > 0)
    eta = Math.round(elapsed / pct * (100 - pct));
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{progress.message || "Indexing..."}</span>
        <span>{pct}%{eta ? ` - ETA ${formatETA(eta)}` : ""}</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-300 ease-out"
          style={{ width: `${pct}%`,
            background: pct < 100 ? "linear-gradient(90deg, #06b6d4, #22d3ee)" : "linear-gradient(90deg, #22c55e, #4ade80)" }}
        />
      </div>
    </div>
  );
}

/* Score breakdown bar */
function ScoreBar({ breakdown }) {
  if (!breakdown) return null;
  const items = [
    { label: "Semantic", key: "semantic_similarity", color: "#06b6d4" },
    { label: "Object Match", key: "object_match", color: "#22d3ee" },
    { label: "Tracking", key: "tracking_consistency", color: "#10b981" },
    { label: "Temporal", key: "temporal_match", color: "#f59e0b" },
    { label: "Motion", key: "motion_activity", color: "#8b5cf6" },
  ].filter(i => (breakdown[i.key] || 0) > 0);
  if (!items.length) return null;
  const total = items.reduce((s, i) => s + Math.max(0, breakdown[i.key] || 0), 0) || 1;
  return (
    <div className="mt-2">
      <div className="flex h-2 rounded-full overflow-hidden bg-gray-700">
        {items.map((i, idx) => (
          <div key={idx} title={`${i.label}: ${((breakdown[i.key] || 0) * 100).toFixed(1)}%`}
            style={{ width: `${Math.max(breakdown[i.key] || 0, 0) / total * 100}%`, background: i.color }} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-1">
        {items.map((i, idx) => (
          <span key={idx} className="text-[10px] text-gray-500">
            <span style={{ color: i.color }}>●</span> {i.label} {((breakdown[i.key] || 0) * 100).toFixed(0)}%
          </span>
        ))}
      </div>
    </div>
  );
}

/* Metric card for dashboard */
function MetricCard({ title, value, unit, icon, color }) {
  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 p-5 hover:border-gray-700 transition-all">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-xl" style={{ color: color || "#06b6d4" }}>{icon || "📊"}</span>
        <span className="text-xs text-gray-500 uppercase tracking-wider">{title}</span>
      </div>
      <div className="text-3xl font-bold text-gray-100">{value}</div>
      {unit && <div className="text-xs text-gray-500 mt-1">{unit}</div>}
    </div>
  );
}

/* Rich result card */
function ResultCard({ seg, index }) {
  const [expanded, setExpanded] = useState(false);
  const vid = seg.video_id;
  const midFrame = seg.frame_indices?.[Math.floor((seg.frame_indices?.length || 1) / 2)];
  const thumbSrc = midFrame != null ? `${API}/api/frames/${vid}/frame_${midFrame}.jpg` : null;
  const detSrc = midFrame != null ? `${API}/api/frames/${vid}/frame_${midFrame}_d.jpg` : null;
  const hasDetections = seg.detections?.length > 0;
  const pct = ((seg.avg_score || seg.weighted_score || 0) * 100).toFixed(1);
  const confidence = parseFloat(pct) > 25 ? "high" : parseFloat(pct) > 15 ? "medium" : "low";
  const confColor = confidence === "high" ? "bg-green-700" : confidence === "medium" ? "bg-yellow-700" : "bg-red-700";
  const startTs = seg.timestamps?.[0];
  const endTs = seg.timestamps?.[seg.timestamps.length - 1];
  const duration = startTs != null && endTs != null ? (endTs - startTs).toFixed(1) : null;

  const copyTs = () => {
    if (startTs != null) {
      navigator.clipboard.writeText(`${startTs.toFixed(1)}s`);
    }
  };

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden hover:border-gray-700 transition-all">
      <div className="flex flex-col sm:flex-row">
        {/* Thumbnail */}
        <div className="relative w-full sm:w-48 h-32 sm:h-auto bg-gray-800 flex-shrink-0">
          {thumbSrc && (
            <img src={detSrc || thumbSrc} alt=""
              className="w-full h-full object-cover"
              onError={(e) => { e.target.style.display = "none" }} />
          )}
          {hasDetections && (
            <span className="absolute top-1 left-1 bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded font-bold">
              AI DETECT
            </span>
          )}
          <span className={cls("absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] font-bold text-white", confColor)}>
            {confidence.toUpperCase()}
          </span>
        </div>
        {/* Details */}
        <div className="flex-1 p-4">
          <div className="flex items-start justify-between mb-1">
            <span className="text-sm font-medium text-gray-200">Result {index + 1}</span>
            <span className="text-xs text-cyan-400 font-medium">{pct}% match</span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-500 mb-2">
            <span>⏱ {startTs?.toFixed(1) || "?"}s{duration ? ` – ${(parseFloat(startTs) + parseFloat(duration)).toFixed(1)}s` : ""}</span>
            {duration && <span>📐 {duration}s duration</span>}
            <span>📄 Frames {seg.frame_indices?.[0]}–{seg.frame_indices?.[seg.frame_indices.length - 1]}</span>
          </div>
          {/* Objects */}
          {hasDetections && (
            <div className="flex flex-wrap gap-1 mb-2">
              {seg.detections.slice(0, 6).map((d, i) => (
                <span key={i} className="text-[10px] bg-gray-800 text-green-400 px-1.5 py-0.5 rounded-full border border-green-900">
                  {d.label} ({Math.round(d.score * 100)}%)
                </span>
              ))}
            </div>
          )}
          {/* Score breakdown */}
          {seg.score_breakdown && (
            <button onClick={() => setExpanded(!expanded)}
              className="text-[10px] text-gray-500 hover:text-cyan-400 mb-1 block">
              {expanded ? "▼ Hide analysis" : "▶ Show score breakdown"}
            </button>
          )}
          {expanded && seg.score_breakdown && <ScoreBar breakdown={seg.score_breakdown} />}
          {/* Explanation */}
          {expanded && (
            <div className="mt-2 text-[11px] text-gray-500 bg-gray-800 rounded p-2">
              <p className="text-gray-400 font-medium mb-1">Why this matched:</p>
              <ul className="space-y-0.5">
                {seg.score_breakdown?.semantic_similarity > 0.1 && <li>✔ Semantic meaning matches your query</li>}
                {seg.detections?.length > 0 && <li>✔ Detected: {seg.detections.map(d => d.label).join(", ")}</li>}
                {seg.score_breakdown?.tracking_consistency > 0 && <li>✔ Object tracked consistently across frames</li>}
                {seg.score_breakdown?.temporal_match > 0 && <li>✔ Inside requested time range</li>}
                {seg.score_breakdown?.motion_activity > 0 && <li>✔ Motion activity detected in segment</li>}
              </ul>
            </div>
          )}
          {/* Actions */}
          <div className="flex gap-2 mt-3">
            {startTs != null && (
              <button onClick={copyTs}
                className="text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded transition-colors">
                📋 Copy Timestamp
              </button>
            )}
            {vid && (
              <a href={`${API}/api/clips/${vid}?start_frame=${seg.frame_indices?.[0] || 0}&end_frame=${seg.frame_indices?.[seg.frame_indices.length - 1] || 0}`}
                className="text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded transition-colors inline-block">
                ⬇ Download Clip
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* App */
function App() {
  const [tab, setTab] = useState("search");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);
  const searchInputRef = useRef(null);

  // Upload state
  const [status, setStatus] = useState("");
  const [logs, setLogs] = useState([]);
  const [indexProgress, setIndexProgress] = useState(null);
  const [indexStartTime, setIndexStartTime] = useState(null);
  const pollingRef = useRef(null);
  const fileRef = useRef(null);

  // Dashboard state
  const [dashboardData, setDashboardData] = useState(null);
  const [timelineData, setTimelineData] = useState(null);
  const [selectedTimelineVideo, setSelectedTimelineVideo] = useState("");
  const [indexedVideos, setIndexedVideos] = useState([]);

  const log = useCallback((msg) => setLogs((p) => [...p, msg]), []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  }, []);

  const pollProgress = useCallback((videoId) => {
    stopPolling();
    pollingRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/videos/${videoId}/index-progress`);
        if (!r.ok) { stopPolling(); setStatus("Error fetching progress"); setIndexProgress(null); return; }
        const p = await r.json();
        setIndexProgress(p);
        if (p.stage === "done" || p.stage === "error" || p.percent >= 100) {
          stopPolling();
          if (p.stage === "error") { setStatus(`Index error: ${p.message}`); log(`Index error: ${p.message}`); }
          else { setStatus(`Ready - ${p.keyframes || "?"} keyframes indexed`); log(`Indexed: ${p.keyframes || "?"} keyframes`); }
          setIndexProgress(null); setIndexStartTime(null);
          fetchIndexedVideos();
        }
      } catch { stopPolling(); setStatus("Progress polling failed"); setIndexProgress(null); }
    }, 800);
  }, [stopPolling, log]);

  const fetchIndexedVideos = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/health`);
      if (r.ok) {
        const d = await r.json();
        setIndexedVideos(d.indexed_videos || []);
      }
    } catch {}
  }, []);

  useEffect(() => { fetchIndexedVideos(); }, [fetchIndexedVideos]);

  const upload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    stopPolling(); setStatus("Uploading..."); setIndexProgress(null); setIndexStartTime(null);
    try {
      const form = new FormData(); form.append("file", file);
      const r = await fetch(`${API}/api/videos/upload`, { method: "POST", body: form });
      if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
      const data = await r.json();
      log(`Uploaded: ${data.filename} (id: ${data.video_id})`);
      setStatus("Starting index...");
      const r2 = await fetch(`${API}/api/videos/${data.video_id}/index`, { method: "POST" });
      if (!r2.ok) throw new Error(`Index start failed: ${r2.status}`);
      setIndexStartTime(Date.now());
      pollProgress(data.video_id);
    } catch (e) { setStatus(`Error: ${e.message}`); log(`Error: ${e.message}`); }
  };

  const doSearch = useCallback(async (q, useV2 = true) => {
    if (!q) return;
    setSearching(true); setResults(null); setSuggestions([]); setHasSearched(true);
    try {
      const endpoint = useV2 ? `/api/v2/search` : `/api/search`;
      const r = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 5, enable_detection: true }),
      });
      if (!r.ok) throw new Error(`Search failed: ${r.status}`);
      const data = await r.json();
      setResults(data);
      log(`Search "${q}": ${data.status} confidence, ${data.segments?.length || 0} segments${data.query_time ? ` (${data.query_time}s)` : ""}`);
    } catch (e) { log(`Search error: ${e.message}`); }
    setSearching(false);
  }, [log]);

  const handleSearch = () => doSearch(query, true);

  const handleInputChange = async (e) => {
    const val = e.target.value;
    setQuery(val);
    if (val.length >= 2) {
      try {
        const r = await fetch(`${API}/api/search/suggest`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: val, top_k: 3 }),
        });
        if (r.ok) { const d = await r.json(); setSuggestions(d.suggestions || []); }
        else setSuggestions([]);
      } catch { setSuggestions([]); }
    } else setSuggestions([]);
  };

  const fetchDashboard = async () => {
    try {
      const r = await fetch(`${API}/api/dashboard/metrics`);
      if (r.ok) { const d = await r.json(); setDashboardData(d); }
    } catch {}
  };

  const fetchTimeline = async (videoId) => {
    if (!videoId) return;
    try {
      const r = await fetch(`${API}/api/videos/${videoId}/timeline`);
      if (r.ok) { const d = await r.json(); setTimelineData(d); }
    } catch {}
  };

  /* Example queries */
  const exampleQueries = [
    "person walking near entrance",
    "red car driving",
    "person carrying backpack",
    "someone picking up an object",
    "person entering and leaving",
  ];

  const tabs = [
    { key: "search", label: "🔍 Search" },
    { key: "upload", label: "📤 Upload" },
    { key: "dashboard", label: "📊 Dashboard" },
    { key: "timeline", label: "⏳ Timeline" },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Top nav */}
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-cyan-400 text-xl font-bold">SceneTrace</span>
            <span className="text-gray-500 text-xs hidden sm:inline">AI</span>
          </div>
          <nav className="flex gap-1">
            {tabs.map(t => (
              <button key={t.key} onClick={() => { stopPolling(); setTab(t.key); if (t.key === "dashboard") fetchDashboard(); if (t.key === "timeline") fetchTimeline(selectedTimelineVideo); }}
                className={cls("px-3 py-1.5 rounded-lg text-sm transition-colors",
                  tab === t.key ? "bg-cyan-600/20 text-cyan-400" : "text-gray-400 hover:text-gray-200 hover:bg-gray-800")}>
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        {/* === SEARCH TAB === */}
        {tab === "search" && (
          <div>
            {/* Hero search area when no search done */}
            <div className={cls("transition-all duration-500", hasSearched ? "mb-6" : "pt-16 pb-8 text-center")}>
              {!hasSearched && (
                <div className="mb-8">
                  <h1 className="text-5xl font-bold text-cyan-400 mb-2">SceneTrace AI</h1>
                  <p className="text-gray-500 text-lg">Describe any event. Find the exact moment.</p>
                </div>
              )}
              <div className={cls("relative", hasSearched ? "max-w-3xl" : "max-w-2xl mx-auto")}>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input ref={searchInputRef} value={query} onChange={handleInputChange}
                      onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                      placeholder={hasSearched ? "Search again..." : 'e.g. "person carrying a red backpack near entrance"'}
                      className="w-full bg-gray-900 border border-gray-700 rounded-xl px-5 py-3.5 text-sm text-gray-100 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-all placeholder-gray-600" />
                    {suggestions.length > 0 && query.length >= 2 && (
                      <div className="absolute top-full mt-1 left-0 right-0 bg-gray-900 border border-gray-700 rounded-xl overflow-hidden shadow-xl z-40">
                        {suggestions.map((s, i) => (
                          <button key={i} onClick={() => { setQuery(s); setSuggestions([]); doSearch(s); }}
                            className="w-full text-left px-4 py-2.5 text-sm text-gray-300 hover:bg-gray-800 hover:text-cyan-400 transition-colors">
                            🔍 {s}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <button onClick={handleSearch} disabled={searching || !query}
                    className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 px-6 py-3.5 rounded-xl font-medium text-sm transition-all">
                    {searching ? (
                      <span className="flex items-center gap-2">
                        <span className="animate-spin inline-block w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full" />
                        Searching
                      </span>
                    ) : "Search"}
                  </button>
                </div>
                {/* Chips */}
                {!hasSearched && (
                  <div className="flex flex-wrap gap-2 mt-4 justify-center">
                    {exampleQueries.map((eq, i) => (
                      <button key={i} onClick={() => { setQuery(eq); doSearch(eq); }}
                        className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded-full border border-gray-700 hover:border-gray-600 transition-all">
                        {eq}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Results */}
            {results && results.segments?.length > 0 && (
              <div className="space-y-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className={cls("px-2.5 py-1 rounded-lg text-xs font-bold",
                    results.status === "high" ? "bg-green-700" : results.status === "medium" ? "bg-yellow-700" : "bg-red-700")}>
                    {results.status.toUpperCase()} CONFIDENCE
                  </span>
                  <span className="text-xs text-gray-500">{results.segments.length} result{results.segments.length > 1 ? "s" : ""}</span>
                  {results.query_time && <span className="text-xs text-gray-600">{(results.query_time * 1000).toFixed(0)}ms</span>}
                </div>
                {results.segments.map((seg, i) => <ResultCard key={i} seg={seg} index={i} />)}
              </div>
            )}
            {results && results.segments?.length === 0 && (
              <div className="text-center py-16">
                <span className="text-5xl mb-4 block">🔍</span>
                <p className="text-gray-500 text-lg mb-2">No confident match found for your description.</p>
                <p className="text-gray-600 text-sm mb-4">Try simplifying your query or describing specific objects.</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {exampleQueries.map((eq, i) => (
                    <button key={i} onClick={() => { setQuery(eq); doSearch(eq); }}
                      className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded-full border border-gray-700">
                      {eq}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* === UPLOAD TAB === */}
        {tab === "upload" && (
          <div className="max-w-xl mx-auto">
            <h2 className="text-xl font-semibold text-gray-200 mb-4">Upload & Index Video</h2>
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
              <input ref={fileRef} type="file" accept="video/*"
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 file:cursor-pointer file:transition-colors mb-4" />
              <button onClick={upload} disabled={!!indexProgress}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 px-6 py-2.5 rounded-xl font-medium text-sm transition-all">
                {indexProgress ? "Indexing..." : "Upload & Index"}
              </button>
              {indexProgress && <ProgressBar progress={indexProgress} startTime={indexStartTime} />}
              {status && !indexProgress && <p className="mt-3 text-sm text-gray-400">{status}</p>}
            </div>
            {logs.length > 0 && (
              <div className="mt-6">
                <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Activity Log</h3>
                <div className="bg-gray-900 rounded-xl border border-gray-800 p-3 text-xs font-mono text-gray-500 max-h-40 overflow-y-auto space-y-1">
                  {logs.map((l, i) => <div key={i}>{l}</div>)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* === DASHBOARD TAB === */}
        {tab === "dashboard" && (
          <div>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-200">Performance Dashboard</h2>
              <button onClick={fetchDashboard}
                className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded-lg transition-colors">
                ⟳ Refresh
              </button>
            </div>
            {dashboardData ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                <MetricCard title="Indexing Speed" value={`${dashboardData.indexing_speed_fps || 0} fps`} unit="frames per second" icon="⚡" color="#22d3ee" />
                <MetricCard title="Videos Indexed" value={dashboardData.videos_indexed || 0} icon="🎬" color="#10b981" />
                <MetricCard title="Total Frames" value={(dashboardData.total_frames || 0).toLocaleString()} icon="📄" color="#8b5cf6" />
                <MetricCard title="Keyframes Kept" value={(dashboardData.total_keyframes || 0).toLocaleString()} icon="🎯" color="#f59e0b" />
                <MetricCard title="Frame Reduction" value={`${dashboardData.avg_frame_reduction_pct || 0}%`} unit="average across all videos" icon="📉" color="#06b6d4" />
                <MetricCard title="Avg Query Latency" value={`${((dashboardData.avg_query_latency || 0) * 1000).toFixed(0)}ms`} unit={`from ${dashboardData.total_queries || 0} queries`} icon="⚡" color="#22d3ee" />
                <MetricCard title="Total Index Time" value={`${dashboardData.total_index_time || 0}s`} unit="cumulative" icon="⏱" color="#f59e0b" />
                <MetricCard title="Avg Embed Time" value={`${dashboardData.avg_embed_time || 0}s`} unit="per video (dominant bottleneck)" icon="🧠" color="#ef4444" />
                {dashboardData.gpu_available !== undefined && (
                  <MetricCard title="GPU" value={dashboardData.gpu_available ? "CUDA ✓" : "CPU"} icon="🖥" color={dashboardData.gpu_available ? "#22c55e" : "#ef4444"} />
                )}
                <MetricCard title="Uptime" value={`${Math.floor((dashboardData.uptime_seconds || 0) / 3600)}h ${Math.floor(((dashboardData.uptime_seconds || 0) % 3600) / 60)}m`} icon="⏰" color="#6366f1" />
              </div>
            ) : (
              <div className="text-center py-16 text-gray-500">
                <p>Loading dashboard data...</p>
                <button onClick={fetchDashboard} className="text-cyan-400 hover:text-cyan-300 mt-2 text-sm">Click to load</button>
              </div>
            )}
          </div>
        )}

        {/* === TIMELINE TAB === */}
        {tab === "timeline" && (
          <div>
            <h2 className="text-xl font-semibold text-gray-200 mb-4">Event Timeline</h2>
            {indexedVideos > 0 ? (
              <div>
                <div className="bg-gray-900 rounded-xl border border-gray-800 p-4 mb-6">
                  <p className="text-xs text-gray-500 mb-3">Select a video to view its frame event timeline:</p>
                  {/* We need video IDs. Use timelineData */}
                </div>
                {timelineData ? (
                  <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="text-sm font-medium text-gray-200">Video: {timelineData.video_id}</h3>
                        <p className="text-xs text-gray-500">
                          {timelineData.metadata?.width}x{timelineData.metadata?.height} · {timelineData.metadata?.fps}fps · {timelineData.metadata?.duration}s · {timelineData.total_keyframes} keyframes
                        </p>
                      </div>
                    </div>
                    {/* Timeline */}
                    <div className="relative overflow-x-auto pb-4">
                      <div className="flex gap-1 min-w-max" style={{ height: 80 }}>
                        {timelineData.events?.map((ev, i) => (
                          <div key={i} className="relative flex flex-col items-center" style={{ width: 12 }}>
                            <div className="w-2 h-2 rounded-full mb-1"
                              style={{
                                background: ev.motion_score > 0.5 ? "#22d3ee" : ev.motion_score > 0.2 ? "#f59e0b" : "#374151",
                                opacity: 0.3 + ev.motion_score
                              }}
                              title={`Frame ${ev.frame_index} @ ${ev.timestamp}s - motion: ${ev.motion_score}`} />
                            <div className="h-full w-px bg-gray-800" />
                            {i % 10 === 0 && (
                              <span className="text-[9px] text-gray-600 mt-1 absolute bottom-0 whitespace-nowrap">
                                {ev.timestamp}s
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                    {/* Frame grid */}
                    <div className="mt-6">
                      <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Keyframes Preview</h4>
                      <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 gap-2">
                        {timelineData.events?.filter(e => e.has_thumbnail).slice(0, 30).map((ev, i) => (
                          <div key={i} className="aspect-video bg-gray-800 rounded overflow-hidden">
                            <img src={`${API}/api/frames/${timelineData.video_id}/frame_${ev.frame_index}.jpg`} alt=""
                              className="w-full h-full object-cover"
                              onError={(e) => { e.target.style.display = "none" }} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>Select a video to view timeline</p>
                    <div className="flex gap-2 justify-center mt-3">
                      {indexedVideos.map((vid) => (
                        <button key={vid} onClick={() => { setSelectedTimelineVideo(vid); fetchTimeline(vid); }}
                          className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded-lg">
                          {vid}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-16 text-gray-500">
                <p>No indexed videos yet. Go to Upload tab to index your first video.</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
