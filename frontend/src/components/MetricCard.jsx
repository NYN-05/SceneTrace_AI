export default function MetricCard({ title, value, unit, icon, color }) {
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
