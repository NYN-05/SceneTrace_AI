export default function ScoreBar({ breakdown }) {
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
