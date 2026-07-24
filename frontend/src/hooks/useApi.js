import { useState, useCallback, useRef, useEffect } from "react";

const API = "";

const MAX_LOG_ENTRIES = 50;

export function useUpload() {
  const [status, setStatus] = useState("");
  const [logs, setLogs] = useState([]);
  const [indexProgress, setIndexProgress] = useState(null);
  const [indexStartTime, setIndexStartTime] = useState(null);
  const [lastVideoId, setLastVideoId] = useState(null);
  const sseRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    return () => {
      if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    };
  }, []);

  const log = useCallback((msg) => setLogs((p) => {
    if (p.length >= MAX_LOG_ENTRIES) return [...p.slice(-(MAX_LOG_ENTRIES - 1)), msg];
    return [...p, msg];
  }), []);

  const stopSSE = useCallback(() => {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
  }, []);

  const connectSSE = useCallback((videoId, onDone) => {
    stopSSE();
    const es = new EventSource(`${API}/api/videos/${videoId}/index-progress/stream`);
    sseRef.current = es;
    es.onmessage = (event) => {
      try {
        const p = JSON.parse(event.data);
        setIndexProgress(p);
        if (p.stage === "done" || p.stage === "error" || p.percent >= 100) {
          es.close(); sseRef.current = null;
          setIndexProgress(null); setIndexStartTime(null);
          if (p.stage === "done") { setStatus(`Ready - ${p.keyframes || "?"} keyframes`); log(`Indexed: ${p.keyframes || "?"} keyframes`); onDone?.(); }
          else { setStatus(`Error: ${p.message}`); log(`Index error: ${p.message}`); }
        }
      } catch {}
    };
    es.onerror = () => {
      es.close(); sseRef.current = null;
      setStatus("Connection lost"); setIndexProgress(null);
    };
  }, [stopSSE, log]);

  const upload = useCallback(async (onDone) => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    stopSSE(); setStatus("Uploading..."); setIndexProgress(null); setIndexStartTime(null);
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
      connectSSE(data.video_id, onDone);
    } catch (e) { setStatus(`Error: ${e.message}`); log(`Error: ${e.message}`); }
  }, [stopSSE, connectSSE, log]);

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
  const abortRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const doSearch = useCallback(async (q, useV2 = true, videoId = null) => {
    if (!q) return;
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setSearching(true); setResults(null); setSuggestions([]); setHasSearched(true);
    try {
      const endpoint = useV2 ? "/api/v2/search" : "/api/search";
      const body = { query: q, top_k: 5, enable_detection: true };
      if (videoId) body.video_id = videoId;
      const r = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!r.ok) throw new Error(`Search failed: ${r.status}`);
      const data = await r.json();
      if (!controller.signal.aborted) setResults(data);
    } catch (e) {
      if (e.name !== "AbortError") console.error(e);
    }
    if (!controller.signal.aborted) setSearching(false);
  }, []);

  const handleInputChange = useCallback((e) => {
    const val = e.target.value;
    setQuery(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (val.length >= 2) {
      debounceRef.current = setTimeout(async () => {
        try {
          const r = await fetch(`${API}/api/search/suggest`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: val, top_k: 3 }),
          });
          if (r.ok) { const d = await r.json(); setSuggestions(d.suggestions || []); }
          else setSuggestions([]);
        } catch { setSuggestions([]); }
      }, 250);
    } else setSuggestions([]);
  }, []);

  return { query, setQuery, results, searching, suggestions, setSuggestions, hasSearched, searchInputRef, doSearch, handleInputChange, setHasSearched };
}
