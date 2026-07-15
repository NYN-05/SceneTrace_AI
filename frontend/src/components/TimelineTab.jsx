const API = "";

export default function TimelineTab({ indexedVideos, timelineData, selectedTimelineVideo, fetchTimeline, setSelectedTimelineVideo }) {
  return (
    <div>
      <h2 className="text-xl font-semibold text-gray-200 mb-4">Event Timeline</h2>
      {indexedVideos.length > 0 ? (
        <div>
          {timelineData ? (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-200">Video: {timelineData.video_id}</h3>
                  <p className="text-xs text-gray-500">
                    {timelineData.metadata?.width}x{timelineData.metadata?.height} · {timelineData.metadata?.fps}fps · {timelineData.metadata?.duration}s · {timelineData.total_keyframes} keyframes
                  </p>
                </div>
              </div>
              <div className="relative overflow-x-auto pb-4">
                <div className="flex gap-1 min-w-max" style={{ height: 80 }}>
                  {timelineData.events?.map((ev, i) => (
                    <div key={i} className="relative flex flex-col items-center" style={{ width: 12 }}>
                      <div className="w-2 h-2 rounded-full mb-1"
                        style={{
                          background: ev.motion_score > 0.5 ? "#22d3ee" : ev.motion_score > 0.2 ? "#f59e0b" : "#374151",
                          opacity: 0.3 + ev.motion_score
                        }}
                        title={`Frame ${ev.frame_index} @ ${ev.timestamp}s - motion: ${ev.motion_score}`} />
                      <div className="h-full w-px bg-gray-800" />
                      {i % 10 === 0 && (
                        <span className="text-[9px] text-gray-600 mt-1 absolute bottom-0 whitespace-nowrap">
                          {ev.timestamp}s
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-6">
                <h4 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Keyframes Preview</h4>
                <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 gap-2">
                  {timelineData.events?.filter(e => e.has_thumbnail).slice(0, 30).map((ev, i) => (
                    <div key={i} className="aspect-video bg-gray-800 rounded overflow-hidden">
                      <img src={`${API}/api/frames/${timelineData.video_id}/frame_${ev.frame_index}.jpg`} alt=""
                        className="w-full h-full object-cover"
                        onError={(e) => { e.target.style.display = "none" }} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>Select a video to view timeline</p>
              <div className="flex gap-2 justify-center mt-3">
                {indexedVideos.map((vid) => (
                  <button key={vid} onClick={() => { setSelectedTimelineVideo(vid); fetchTimeline(vid); }}
                    className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded-lg">
                    {vid}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-16 text-gray-500">
          <p>No indexed videos yet. Go to Upload tab to index your first video.</p>
        </div>
      )}
    </div>
  );
}
