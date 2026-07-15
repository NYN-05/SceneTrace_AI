import { useState } from "react";
import ScoreBar from "./ScoreBar";

const API = "";

function cls(...args) { return args.filter(Boolean).join(" "); }

export default function ResultCard({ seg, index }) {
  const [expanded, setExpanded] = useState(false);
  const vid = seg.video_id;
  const midFrame = seg.frame_indices?.[Math.floor((seg.frame_indices?.length || 1) / 2)];
  const thumbSrc = midFrame != null ? `${API}/api/frames/${vid}/frame_${midFrame}.jpg` : null;
  const detSrc = midFrame != null ? `${API}/api/frames/${vid}/frame_${midFrame}_d.jpg` : null;
  const hasDetections = seg.detections?.length > 0;
  const pct = ((seg.avg_score || seg.weighted_score || 0) * 100).toFixed(1);
  const confidence = parseFloat(pct) > 25 ? "high" : parseFloat(pct) > 15 ? "medium" : "low";
  const confColor = confidence === "high" ? "bg-green-700" : confidence === "medium" ? "bg-yellow-700" : "bg-red-700";
  const startTs = seg.timestamps?.[0];
  const endTs = seg.timestamps?.[seg.timestamps.length - 1];
  const duration = startTs != null && endTs != null ? (endTs - startTs).toFixed(1) : null;

  const copyTs = () => {
    if (startTs != null) navigator.clipboard.writeText(`${startTs.toFixed(1)}s`);
  };

  return (
    <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden hover:border-gray-700 transition-all">
      <div className="flex flex-col sm:flex-row">
        <div className="relative w-full sm:w-48 h-32 sm:h-auto bg-gray-800 flex-shrink-0">
          {thumbSrc && (
            <img src={hasDetections ? detSrc : thumbSrc} alt=""           
              className="w-full h-full object-cover"
              onError={(e) => { e.target.style.display = "none" }} />
          )}
          {hasDetections && (
            <span className="absolute top-1 left-1 bg-green-600 text-white text-[10px] px-1.5 py-0.5 rounded font-bold">
              AI DETECT
            </span>
          )}
          <span className={cls("absolute bottom-1 right-1 px-1.5 py-0.5 rounded text-[10px] font-bold text-white", confColor)}>
            {confidence.toUpperCase()}
          </span>
        </div>
        <div className="flex-1 p-4">
          <div className="flex items-start justify-between mb-1">
            <span className="text-sm font-medium text-gray-200">Result {index + 1}</span>
            <span className="text-xs text-cyan-400 font-medium">{pct}% match</span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-500 mb-2">
            <span>⏱ {startTs?.toFixed(1) || "?"}s{duration ? ` – ${(parseFloat(startTs) + parseFloat(duration)).toFixed(1)}s` : ""}</span>
            {duration && <span>📐 {duration}s duration</span>}
            <span>📄 Frames {seg.frame_indices?.[0]}–{seg.frame_indices?.[seg.frame_indices.length - 1]}</span>
          </div>
          {hasDetections && (
            <div className="flex flex-wrap gap-1 mb-2">
              {seg.detections.slice(0, 6).map((d, i) => (
                <span key={i} className="text-[10px] bg-gray-800 text-green-400 px-1.5 py-0.5 rounded-full border border-green-900">
                  {d.label} ({Math.round(d.score * 100)}%)
                </span>
              ))}
            </div>
          )}
          {seg.score_breakdown && (
            <button onClick={() => setExpanded(!expanded)}
              className="text-[10px] text-gray-500 hover:text-cyan-400 mb-1 block">
              {expanded ? "▼ Hide analysis" : "▶ Show score breakdown"}
            </button>
          )}
          {expanded && seg.score_breakdown && <ScoreBar breakdown={seg.score_breakdown} />}
          {expanded && (
            <div className="mt-2 text-[11px] text-gray-500 bg-gray-800 rounded p-2">
              <p className="text-gray-400 font-medium mb-1">Why this matched:</p>
              <ul className="space-y-0.5">
                {seg.score_breakdown?.semantic_similarity > 0.1 && <li>✔ Semantic meaning matches your query</li>}
                {seg.detections?.length > 0 && <li>✔ Detected: {seg.detections.map(d => d.label).join(", ")}</li>}
                {seg.score_breakdown?.tracking_consistency > 0 && <li>✔ Object tracked consistently</li>}
                {seg.score_breakdown?.temporal_match > 0 && <li>✔ Inside requested time range</li>}
                {seg.score_breakdown?.motion_activity > 0 && <li>✔ Motion activity detected</li>}
              </ul>
            </div>
          )}
          <div className="flex gap-2 mt-3">
            {startTs != null && (
              <button onClick={copyTs}
                className="text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded transition-colors">
                📋 Copy Timestamp
              </button>
            )}
            {vid && (
              <a href={`${API}/api/clips/${vid}?start_frame=${seg.frame_indices?.[0] || 0}&end_frame=${seg.frame_indices?.[seg.frame_indices.length - 1] || 0}`}
                className="text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2.5 py-1 rounded transition-colors inline-block">
                ⬇ Download Clip
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
