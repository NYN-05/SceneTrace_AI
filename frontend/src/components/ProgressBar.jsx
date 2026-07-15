function formatETA(sec) {
  if (sec == null || sec <= 0) return "";
  if (sec < 60) return `~${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s > 0 ? `~${m}m ${s}s` : `~${m}m`;
}

export default function ProgressBar({ progress, startTime }) {
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
