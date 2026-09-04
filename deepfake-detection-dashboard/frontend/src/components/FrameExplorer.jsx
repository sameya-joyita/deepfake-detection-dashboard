import { useMemo, useState } from "react";

import { formatNumber, formatScore } from "../lib/format";

export default function FrameExplorer({ frames = [] }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selected = frames[selectedIndex] || null;
  const scores = useMemo(() => frames.map((frame) => frame.dual_score), [frames]);

  if (!frames.length || !selected) return null;

  return (
    <section className="panel frame-panel" aria-labelledby="frames-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Frame evidence</span>
          <h2 id="frames-title">Sampled faces and score variation</h2>
        </div>
        <span className="muted-label">{frames.length} usable frames</span>
      </div>

      <div className="frame-chart" aria-label="Dual model score by sampled face">
        {scores.map((score, index) => (
          <button
            key={`${frames[index].video_frame_index}-${index}`}
            className={selectedIndex === index ? "active" : ""}
            style={{ height: `${Math.max(6, (score || 0) * 100)}%` }}
            onClick={() => setSelectedIndex(index)}
            title={`Frame ${frames[index].video_frame_index}: ${formatScore(score)}`}
            aria-label={`Inspect sampled frame ${index + 1}`}
          />
        ))}
        <span className="chart-boundary" aria-hidden="true" />
      </div>

      <div className="frame-detail">
        <img src={selected.preview_data_url} alt={`Sampled video frame ${selected.video_frame_index} with detected face box`} />
        <div>
          <span className="section-kicker">Selected sample {selectedIndex + 1}</span>
          <h3>Video frame {selected.video_frame_index}</h3>
          <dl className="detail-list">
            <div><dt>Dual score</dt><dd>{formatScore(selected.dual_score)}</dd></div>
            <div><dt>Spatial score</dt><dd>{formatScore(selected.spatial_score)}</dd></div>
            <div><dt>Gate alpha</dt><dd>{formatNumber(selected.gate_alpha, 3)}</dd></div>
            <div><dt>Face confidence</dt><dd>{formatScore(selected.face_confidence)}</dd></div>
          </dl>
          <img className="face-crop" src={selected.face_crop_data_url} alt={`Extracted face crop from frame ${selected.video_frame_index}`} />
        </div>
      </div>
    </section>
  );
}

