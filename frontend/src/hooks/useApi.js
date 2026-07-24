import { useState, useCallback, useRef, useEffect } from "react";

const API = "";

export function useUpload() {
  const [status, setStatus] = useState("");
  const [logs, setLogs] = useState([]);
  const [indexProgress, setIndexProgress] = useState(null);
  const [indexStartTime, setIndexStartTime] = useState(null);
  const [lastVideoId, setLastVideoId] = useState(null);
  const pollingRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
    };
  }, []);

  const log = useCallback((msg) => setLogs((p) => [...p, msg]), []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  }, []);

  const pollProgress = useCallback((videoId, onDone) => {
    stopPolling();
    pollingRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/api/videos/${videoId}/index-progress`);
        if (!r.ok) { stopPolling(); setStatus("Error"); setIndexProgress(null); return; }
        const p = await r.json();
        setIndexProgress(p);
        if (p.stage === "done" || p.stage === "error" || p.percent >= 100) {
          stopPolling();
          setIndexProgress(null); setIndexStartTime(null);
          if (p.stage === "done") { setStatus(`Ready - ${p.keyframes || "?"} keyframes`); log(`Indexed: ${p.keyframes || "?"} keyframes`); onDone?.(); }
          else { setStatus(`Error: ${p.message}`); log(`Index error: ${p.message}`); }
        }
      } catch { stopPolling(); setStatus("Polling failed"); setIndexProgress(null); }
    }, 800);
  }, [stopPolling, log]);

  const upload = useCallback(async (onDone) => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    stopPolling(); setStatus("Uploading..."); setIndexProgress(null); setIndexStartTime(null);
    try {
      const form = new FormData(); form.append("file", file);
      const r = await fetch(`${API}/api/videos/upload`, { method: "POST", body: form });
      if (!r.ok) throw new Error(`Upload failed: ${r.status}`);
      const data = await r.json();
      log(`Uploaded: ${data.filename} (${data.video_id})`);
      setLastVideoId(data.video_id);
      setStatus("Indexing...");
      const r2 = await fetch(`${API}/api/videos/${data.video_id}/index`, { method: "POST" });
      if (!r2.ok) throw new Error(`Index start failed: ${r2.status}`);
      setIndexStartTime(Date.now());
      pollProgress(data.video_id, onDone);
    } catch (e) { setStatus(`Error: ${e.message}`); log(`Error: ${e.message}`); }
  }, [stopPolling, pollProgress, log]);

  const clearLastVideo = useCallback(() => setLastVideoId(null), []);

  return { status, logs, indexProgress, indexStartTime, lastVideoId, clearLastVideo, fileRef, upload, setLogs, setStatus };
}

export function useSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);
  const searchInputRef = useRef(null);

  const doSearch = useCallback(async (q, useV2 = true, videoId = null) => {
    if (!q) return;
    setSearching(true); setResults(null); setSuggestions([]); setHasSearched(true);
    try {
      const endpoint = useV2 ? "/api/v2/search" : "/api/search";
      const body = { query: q, top_k: 5, enable_detection: true };
      if (videoId) body.video_id = videoId;
      const r = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`Search failed: ${r.status}`);
      const data = await r.json();
      setResults(data);
    } catch (e) { console.error(e); }
    setSearching(false);
  }, []);

  const handleInputChange = useCallback(async (e) => {
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
  }, []);

  return { query, setQuery, results, searching, suggestions, setSuggestions, hasSearched, searchInputRef, doSearch, handleInputChange, setHasSearched };
}
