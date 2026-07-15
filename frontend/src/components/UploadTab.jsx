import ProgressBar from "./ProgressBar";

export default function UploadTab({ status, logs, indexProgress, indexStartTime, fileRef, upload, setLogs }) {
  return (
    <div className="max-w-xl mx-auto">
      <h2 className="text-xl font-semibold text-gray-200 mb-4">Upload & Index Video</h2>
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
        <input ref={fileRef} type="file" accept="video/*"
          className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:bg-cyan-600 file:text-white hover:file:bg-cyan-500 file:cursor-pointer file:transition-colors mb-4" />
        <button onClick={upload} disabled={!!indexProgress}
          className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:text-gray-500 px-6 py-2.5 rounded-xl font-medium text-sm transition-all">
          {indexProgress ? "Indexing..." : "Upload & Index"}
        </button>
        {indexProgress && <ProgressBar progress={indexProgress} startTime={indexStartTime} />}
        {status && !indexProgress && <p className="mt-3 text-sm text-gray-400">{status}</p>}
      </div>
      {logs.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-2">Activity Log</h3>
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-3 text-xs font-mono text-gray-500 max-h-40 overflow-y-auto space-y-1">
            {logs.map((l, i) => <div key={i}>{l}</div>)}
          </div>
        </div>
      )}
    </div>
  );
}
