import ResultCard from "./ResultCard";

function cls(...args) { return args.filter(Boolean).join(" "); }

const exampleQueries = [
  "person walking near entrance", "red car driving",
  "person carrying backpack", "someone picking up an object",
  "person entering and leaving",
];

export default function SearchTab({ query, setQuery, results, searching, suggestions, hasSearched, searchInputRef, doSearch, handleInputChange, setHasSearched }) {
  const handleSearch = () => doSearch(query, true);

  return (
    <div>
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
              <button key={i} onClick={() => { setQuery(eq); doSearch(eq); }}
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
