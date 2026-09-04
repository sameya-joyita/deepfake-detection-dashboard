import { formatScore } from "../lib/format";

function ScoreMeter({ label, score, colour, detail }) {
  const safeScore = Number.isFinite(score) ? Math.min(1, Math.max(0, score)) : 0;
  return (
    <div className="score-row">
      <div className="score-row-heading">
        <div>
          <strong>{label}</strong>
          <span>{detail}</span>
        </div>
        <b>{formatScore(score)}</b>
      </div>
      <div className="score-track" aria-label={`${label} score ${formatScore(score)}`}>
        <span className="score-boundary" aria-hidden="true" />
        <span
          className="score-fill"
          style={{ width: `${safeScore * 100}%`, backgroundColor: colour }}
        />
      </div>
      <div className="score-scale"><span>Likely authentic</span><span>0.5 boundary</span><span>Likely manipulated</span></div>
    </div>
  );
}

export default function ScoreComparison({ result }) {
  const spatial = result?.scores?.spatial_score;
  const dual = result?.scores?.dual_score;
  const agreement = result?.model_comparison?.agreement;

  if (!result?.scores) return null;

  return (
    <section className="panel" aria-labelledby="score-comparison-title">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Model comparison</span>
          <h2 id="score-comparison-title">Video-level manipulation scores</h2>
        </div>
        <span className={`agreement-badge ${agreement ? "agrees" : "disagrees"}`}>
          {agreement ? "Models agree" : "Models disagree"}
        </span>
      </div>

      <div className="score-stack">
        <ScoreMeter
          label="Spatial baseline"
          score={spatial}
          colour="#53718f"
          detail="EfficientNet-B4 facial appearance evidence"
        />
        <ScoreMeter
          label="Spatial-frequency model"
          score={dual}
          colour="#0c8d83"
          detail="Official gated fusion checkpoint"
        />
      </div>

      <p className="context-note">
        These are raw video scores obtained by averaging frame sigmoid outputs.
        They are not calibrated probabilities. The 0.5 line is the fixed class
        boundary used in the notebook.
      </p>
    </section>
  );
}

