import { memo, useState } from "react";
import ResultCard from "./ResultCard";
import ProgressBar from "./ProgressBar";

function cls(...args) { return args.filter(Boolean).join(" "); }

const LogItem = memo(function LogItem({ msg }) {
  return <div>{msg}</div>;
});

const exampleQueries = [
  "person walking near entrance", "red car driving",
  "person carrying backpack", "someone picking up an object",
  "person entering and leaving",
];

export default function SearchPage({
  query, setQuery, results, searching, suggestions, hasSearched,
  searchInputRef, doSearch, handleInputChange, setHasSearched,
  uploadStatus, uploadLogs, indexProgress, indexStartTime,
  fileRef, upload, lastVideoId, clearLastVideo, onUploadDone,
}) {
  const [showUpload, setShowUpload] = useState(false);

  const handleSearch = () => doSearch(query, true, lastVideoId);

  const handleExampleClick = (eq) => {
    setQuery(eq);
    doSearch(eq, true, lastVideoId);
  };

  return (
    <div>
      {/* Upload section */}
      <div className="max-w-2xl mx-auto mb-6">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">
              {lastVideoId ? (
                <span className="text-green-400">Active video: <code className="text-xs">{lastVideoId}</code></span>
              ) : "Upload a video to search"}
            </h3>
            <div className="flex gap-2">
              {lastVideoId && (
                <button onClick={clearLastVideo}
                  className="text-xs text-gray-500 hover:text-red-400 px-2 py-1 rounded transition-colors">
                  Clear
                </button>
              )}
              <button onClick={() => setShowUpload(!showUpload)}
                className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1 rounded-lg transition-colors">
                {showUpload ? "Hide" : lastVideoId ? "Change Video" : "Browse"}
              </button>
            </div>
          </div>
          {showUpload && (
            <div>
              <input ref={fileRef} type="file" accept="video/*"
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 file:cursor-pointer file:transition-colors mb-3" />
              <button onClick={() => upload(onUploadDone)} disabled={!!indexProgress}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 px-5 py-2 rounded-xl font-medium text-sm transition-all">
                {indexProgress ? "Indexing..." : "Upload & Index"}
              </button>
              {indexProgress && <ProgressBar progress={indexProgress} startTime={indexStartTime} />}
              {uploadStatus && !indexProgress && <p className="mt-2 text-sm text-gray-400">{uploadStatus}</p>}
            </div>
          )}
          {uploadLogs.length > 0 && (
            <div className="mt-2 text-xs font-mono text-gray-600 max-h-16 overflow-y-auto space-y-0.5">
              {uploadLogs.map((l, i) => <LogItem key={i} msg={l} />)}
            </div>
          )}
        </div>
      </div>

      {/* Search section */}
      <div className={cls("transition-all duration-500", hasSearched ? "mb-6" : "pt-8 pb-8 text-center")}>
        {!hasSearched && !lastVideoId && (
          <div className="mb-6">
            <h1 className="text-4xl font-bold text-cyan-400 mb-1">SceneTrace AI</h1>
            <p className="text-gray-500 text-base">Upload a video, then describe what to find.</p>
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
                    <button key={i} onClick={() => { setQuery(s); setSuggestions([]); doSearch(s, true, lastVideoId); }}
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
          {!hasSearched && (
            <div className="flex flex-wrap gap-2 mt-4 justify-center">
              {exampleQueries.map((eq, i) => (
                <button key={i} onClick={() => handleExampleClick(eq)}
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
          <p className="text-gray-500 text-lg mb-2">No confident match found.</p>
          <p className="text-gray-600 text-sm mb-4">Try simplifying your query or describing specific objects.</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {exampleQueries.map((eq, i) => (
              <button key={i} onClick={() => handleExampleClick(eq)}
                className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded-full border border-gray-700">
                {eq}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}