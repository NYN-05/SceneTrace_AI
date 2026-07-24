import { useState, useCallback, useEffect } from "react";
import SearchPage from "./components/SearchPage";
import DashboardTab from "./components/DashboardTab";
import TimelineTab from "./components/TimelineTab";
import { useUpload, useSearch } from "./hooks/useApi";

function cls(...args) { return args.filter(Boolean).join(" "); }

const API = "";

export default function App() {
  const [tab, setTab] = useState("search");
  const [indexedVideos, setIndexedVideos] = useState([]);
  const [dashboardData, setDashboardData] = useState(null);
  const [timelineData, setTimelineData] = useState(null);
  const [selectedTimelineVideo, setSelectedTimelineVideo] = useState("");

  const search = useSearch();
  const upload = useUpload();

  const fetchIndexedVideos = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/health`);
      if (r.ok) { const d = await r.json(); setIndexedVideos(d.video_ids || []); }
    } catch {}
  }, []);

  const fetchDashboard = useCallback(async () => {
    try {
      const r = await fetch(`${API}/api/dashboard/metrics`);
      if (r.ok) { const d = await r.json(); setDashboardData(d); }
    } catch {}
  }, []);

  const fetchTimeline = useCallback(async (videoId) => {
    if (!videoId) return;
    try {
      const r = await fetch(`${API}/api/videos/${videoId}/timeline`);
      if (r.ok) { const d = await r.json(); setTimelineData(d); }
    } catch {}
  }, []);

  useEffect(() => { fetchIndexedVideos(); }, [fetchIndexedVideos]);

  const tabs = [
    { key: "search", label: "Search" },
    { key: "dashboard", label: "Dashboard" },
    { key: "timeline", label: "Timeline" },
  ];

  const onUploadDone = useCallback(() => {
    fetchIndexedVideos();
  }, [fetchIndexedVideos]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-cyan-400 text-xl font-bold">SceneTrace</span>
            <span className="text-gray-500 text-xs hidden sm:inline">AI</span>
          </div>
          <nav className="flex gap-1">
            {tabs.map(t => (
              <button key={t.key} onClick={() => { setTab(t.key); if (t.key === "dashboard") fetchDashboard(); if (t.key === "timeline") fetchTimeline(selectedTimelineVideo); }}
                className={cls("px-3 py-1.5 rounded-lg text-sm transition-colors",
                  tab === t.key ? "bg-cyan-600/20 text-cyan-400" : "text-gray-400 hover:text-gray-200 hover:bg-gray-800")}>
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 py-6">
        {tab === "search" && (
          <SearchPage
            {...search}
            uploadStatus={upload.status}
            uploadLogs={upload.logs}
            indexProgress={upload.indexProgress}
            indexStartTime={upload.indexStartTime}
            fileRef={upload.fileRef}
            lastVideoId={upload.lastVideoId}
            clearLastVideo={upload.clearLastVideo}
            upload={upload.upload}
            onUploadDone={onUploadDone}
          />
        )}
        {tab === "dashboard" && <DashboardTab dashboardData={dashboardData} fetchDashboard={fetchDashboard} />}
        {tab === "timeline" && (
          <TimelineTab
            indexedVideos={indexedVideos}
            timelineData={timelineData}
            selectedTimelineVideo={selectedTimelineVideo}
            fetchTimeline={fetchTimeline}
            setSelectedTimelineVideo={setSelectedTimelineVideo}
          />
        )}
      </main>
    </div>
  );
}