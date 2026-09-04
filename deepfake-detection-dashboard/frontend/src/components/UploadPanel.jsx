import { useRef, useState } from "react";

import { UploadIcon } from "./Icons";
import { formatBytes } from "../lib/format";

const ACCEPTED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"];

function isAccepted(file) {
  const name = file?.name?.toLowerCase() || "";
  return ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension));
}

export default function UploadPanel({ busy, onAnalyze }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [includeGradcam, setIncludeGradcam] = useState(true);
  const [localError, setLocalError] = useState("");

  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (!isAccepted(nextFile)) {
      setLocalError("Choose an MP4, MOV, AVI or MKV video.");
      setFile(null);
      return;
    }
    if (nextFile.size > 250 * 1024 * 1024) {
      setLocalError("The video exceeds the 250 MB upload limit.");
      setFile(null);
      return;
    }
    setLocalError("");
    setFile(nextFile);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  function submit(event) {
    event.preventDefault();
    if (!file || busy) return;
    onAnalyze(file, includeGradcam);
  }

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="section-kicker">New analysis</div>
      <div className="upload-heading-row">
        <div>
          <h1 id="upload-title">Examine a face-centred video</h1>
          <p>
            The pipeline samples up to 20 frames, extracts one face per frame
            and compares spatial with spatial-frequency evidence.
          </p>
        </div>
        <span className="protocol-chip">Locked 20-frame protocol</span>
      </div>

      <form onSubmit={submit}>
        <div
          className={`drop-zone ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
          }}
          role="button"
          tabIndex="0"
          aria-label="Choose a video to analyse"
        >
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept=".mp4,.mov,.avi,.mkv,video/mp4,video/quicktime,video/x-msvideo"
            onChange={(event) => chooseFile(event.target.files?.[0])}
          />
          <span className="upload-icon"><UploadIcon /></span>
          {file ? (
            <div>
              <strong>{file.name}</strong>
              <span>{formatBytes(file.size)} · click to replace</span>
            </div>
          ) : (
            <div>
              <strong>Drop a video here or browse</strong>
              <span>MP4, MOV, AVI or MKV · maximum 250 MB</span>
            </div>
          )}
        </div>

        {localError && <p className="field-error" role="alert">{localError}</p>}

        <div className="upload-actions">
          <label className="check-control">
            <input
              type="checkbox"
              checked={includeGradcam}
              onChange={(event) => setIncludeGradcam(event.target.checked)}
            />
            <span>Generate Grad-CAM explanation</span>
          </label>
          <button className="primary-button" type="submit" disabled={!file || busy}>
            {busy ? "Analysing video…" : "Run analysis"}
          </button>
        </div>
      </form>
    </section>
  );
}

