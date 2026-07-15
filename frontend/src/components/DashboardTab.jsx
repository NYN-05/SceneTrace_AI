import MetricCard from "./MetricCard";

export default function DashboardTab({ dashboardData, fetchDashboard }) {
  return (
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
  );
}
