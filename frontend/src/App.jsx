import React, { useState, useRef, useCallback } from "react";

const API = "";

function formatETA(sec) {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `~${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s > 0 ? `~${m}m ${s}s` : `~${m}m`;
}

function ProgressBar({ progress, startTime }) {
  if (!progress) return null;
  const pct = progress.percent || 0;
  const elapsed = startTime ? (Date.now() - startTime) / 1000 : 0;
  let eta = progress.eta_seconds;
  if (eta == null && pct > 0 && pct < 100 && elapsed > 0) {
    eta = Math.round(elapsed / pct * (100 - pct));
  }
  return (
    <div className="mt-3">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{progress.message || "Indexing..."}</span>
        <span>{pct}%{eta ? ` - ETA ${formatETA(eta)}` : ""}</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300 ease-out"
          style={{
            width: `${pct}%`,
            background: pct < 100
              ? "linear-gradient(90deg, #06b6d4, #22d3ee)"
              : "linear-gradient(90deg, #22c55e, #4ade80)"
          }}
        />
      </div>
    </div>
  );
}

function App() {
  const [tab, setTab] = useState("upload");
  const [status, setStatus] = useState("");
  const [logs, setLogs] = useState([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [indexProgress, setIndexProgress] = useState(null);
  const [indexStartTime, setIndexStartTime] = useState(null);
  const pollingRef = useRef(null);
  const fileRef = useRef(null);

  const log = useCallback((msg) => setLogs((p) => [...p, msg]), []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollProgress = useCallback((videoId) => {
    stopPolling();
    pollingRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/videos/${videoId}/index-progress`);
        if (!r.ok) {
          stopPolling();
          setStatus("Error fetching progress");
          setIndexProgress(null);
          return;
        }
        const p = await r.json();
        setIndexProgress(p);
        if (p.stage === "done" || p.stage === "error" || p.percent >= 100) {
          stopPolling();
          if (p.stage === "error") {
            setStatus(`Index error: ${p.message}`);
            log(`Index error: ${p.message}`);
          } else {
            setStatus(`Ready - ${p.keyframes || "?"} keyframes indexed`);
            log(`Indexed: ${p.keyframes || "?"} keyframes`);
          }
          setIndexProgress(null);
          setIndexStartTime(null);
        }
      } catch (e) {
        stopPolling();
        setStatus("Progress polling failed");
        setIndexProgress(null);
      }
    }, 800);
  }, [stopPolling, log]);

  const upload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    stopPolling();
    setStatus("Uploading...");
    setIndexProgress(null);
    setIndexStartTime(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch(`${API}/api/videos/upload`, { method: "POST", body: form });
      if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
      const data = await r.json();
      log(`Uploaded: ${data.filename} (id: ${data.video_id})`);
      setStatus("Starting index...");
      const r2 = await fetch(`${API}/api/videos/${data.video_id}/index`, { method: "POST" });
      if (!r2.ok) throw new Error(`Index start failed: ${r2.status}`);
      setIndexStartTime(Date.now());
      pollProgress(data.video_id);
    } catch (e) {
      setStatus(`Error: ${e.message}`);
      log(`Error: ${e.message}`);
    }
  };

  const search = async () => {
    if (!query) return;
    setSearching(true);
    setResults(null);
    try {
      const r = await fetch(`${API}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!r.ok) throw new Error(`Search failed: ${r.status}`);
      const data = await r.json();
      setResults(data);
      log(`Search "${query}": ${data.status} confidence, ${data.segments?.length || 0} segments`);
    } catch (e) {
      log(`Search error: ${e.message}`);
    }
    setSearching(false);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 max-w-6xl mx-auto">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-cyan-400">SceneTrace AI</h1>
        <p className="text-gray-400 text-sm mt-1">Natural Language Video Grounding</p>
      </header>

      <div className="flex gap-2 mb-6">
        <button onClick={() => setTab("upload")} className={`px-4 py-2 rounded ${tab === "upload" ? "bg-cyan-600" : "bg-gray-800"}`}>Upload</button>
        <button onClick={() => { stopPolling(); setTab("search"); }} className={`px-4 py-2 rounded ${tab === "search" ? "bg-cyan-600" : "bg-gray-800"}`}>Search</button>
      </div>

      {tab === "upload" && (
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
          <input ref={fileRef} type="file" accept="video/*" className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 mb-4" />
          <button onClick={upload} disabled={!!indexProgress} className="bg-cyan-600 hover:bg-cyan-500 px-6 py-2 rounded font-medium disabled:opacity-50">Upload & Index</button>
          {indexProgress && <ProgressBar progress={indexProgress} startTime={indexStartTime} />}
          {status && !indexProgress && <p className="mt-3 text-sm text-gray-300">{status}</p>}
        </div>
      )}

      {tab === "search" && (
        <div className="space-y-6">
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="flex gap-3">
              <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && search()} placeholder='e.g., "Find a person picking up a backpack"' className="flex-1 bg-gray-800 border border-gray-700 rounded px-4 py-2 text-sm focus:outline-none focus:border-cyan-500" />
              <button onClick={search} disabled={searching} className="bg-cyan-600 hover:bg-cyan-500 px-6 py-2 rounded font-medium disabled:opacity-50">{searching ? "Searching..." : "Search"}</button>
            </div>
          </div>

          {results && (
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center gap-2 mb-4">
                <span className={`px-2 py-0.5 rounded text-xs font-bold ${results.status === "high" ? "bg-green-700" : results.status === "medium" ? "bg-yellow-700" : "bg-red-700"}`}>{results.status.toUpperCase()}</span>
                {results.query_info?.time_range && <span className="text-xs text-gray-400">Time: {results.query_info.time_range}</span>}
              </div>

              {results.segments?.length > 0 ? (
                <div className="space-y-4">
                  {results.segments.map((seg, i) => (
                    <div key={i} className="bg-gray-800 rounded p-4 border border-gray-700">
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-sm font-medium">Segment {i + 1}</span>
                        <span className="text-xs text-cyan-400">{(seg.avg_score * 100).toFixed(1)}% match</span>
                      </div>
                      <div className="flex gap-4 text-xs text-gray-400">
                        <span>Frames: {seg.frame_indices?.[0]}–{seg.frame_indices?.[seg.frame_indices.length - 1]}</span>
                        {seg.timestamps?.[0] && <span>Time: {seg.timestamps[0].toFixed(1)}s</span>}
                      </div>
                      <div className="mt-2 flex gap-1">
                        {seg.frame_indices?.slice(0, 4).map((fi, j) => (
                          <img key={j} src={`${API}/api/frames/${seg.video_id}/frame_${fi}.jpg`} alt="" className="w-20 h-16 object-cover rounded border border-gray-600 bg-gray-700" onError={(e) => { e.target.style.display = "none" }} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No confident match found. Try a simpler description.</p>
              )}
            </div>
          )}
        </div>
      )}

      {logs.length > 0 && (
        <div className="mt-8">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Activity Log</h3>
          <div className="bg-gray-900 rounded p-3 border border-gray-800 text-xs font-mono text-gray-400 max-h-32 overflow-y-auto space-y-1">
            {logs.map((l, i) => <div key={i}>{l}</div>)}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
