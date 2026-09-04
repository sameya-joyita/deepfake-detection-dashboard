import { formatMilliseconds, formatNumber, formatScore } from "../lib/format";

function Metric({ label, value, note }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}

export function FrequencyEvidence({ result }) {
  if (!result?.frequency_counterfactual) return null;
  const effect = result.frequency_counterfactual.score_effect;
  const direction = effect > 0 ? "increased" : effect < 0 ? "decreased" : "did not change";

  return (
    <section className="panel evidence-panel" aria-labelledby="frequency-title">
      <span className="section-kicker">Mechanistic check</span>
      <h2 id="frequency-title">Frequency-residual counterfactual</h2>
      <div className="metric-grid three">
        <Metric label="Official dual score" value={formatScore(result.scores.dual_score)} />
        <Metric label="Residual disabled" value={formatScore(result.scores.frequency_disabled_score)} />
        <Metric
          label="Score effect"
          value={`${effect >= 0 ? "+" : ""}${formatNumber(effect, 3)}`}
          note={`Frequency residual ${direction} the fake score`}
        />
      </div>
      <p>{result.narrative?.frequency_counterfactual}</p>
      <div className="gate-strip">
        <div>
          <span>Mean gate alpha</span>
          <strong>{formatNumber(result.gate_alpha, 3)}</strong>
        </div>
        <p>{result.narrative?.gate_alpha}</p>
      </div>
      <p className="context-note">
        The disabled result uses the same trained checkpoint and classifier with
        the learned frequency residual removed. It is evidence about this
        architecture, not standalone proof that frequency information caused a
        manipulation.
      </p>
    </section>
  );
}

export function InputEvidence({ result }) {
  const input = result?.input_adequacy;
  if (!input) return null;

  return (
    <section className="panel" aria-labelledby="input-title">
      <span className="section-kicker">Input adequacy</span>
      <h2 id="input-title">What the pipeline could inspect</h2>
      <div className="metric-grid">
        <Metric label="Frames requested" value={input.requested_frames ?? "—"} />
        <Metric label="Usable faces" value={input.usable_frames ?? 0} />
        <Metric label="Face coverage" value={formatScore(input.usable_fraction)} />
        <Metric label="Mean brightness" value={formatNumber(input.mean_brightness, 1)} />
      </div>
      <p>{result.narrative?.input_adequacy}</p>
      {input.warnings?.length > 0 && (
        <ul className="warning-list">
          {input.warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      )}
      <p className="context-note">
        Blur and brightness checks are descriptive display heuristics. They do
        not alter the prediction or manual-review rule.
      </p>
    </section>
  );
}

export function ProcessingEvidence({ result }) {
  const processing = result?.processing;
  if (!processing) return null;
  const forward = processing.model_forward_ms;

  return (
    <section className="panel" aria-labelledby="processing-title">
      <span className="section-kicker">Processing record</span>
      <h2 id="processing-title">Pipeline timing</h2>
      <div className="metric-grid">
        <Metric label="Full pipeline" value={formatMilliseconds(processing.total_pipeline_ms)} />
        <Metric label="Preprocessing" value={formatMilliseconds(processing.preprocessing_ms)} />
        <Metric label="Spatial forward" value={formatMilliseconds(forward?.spatial_forward_batch)} />
        <Metric label="Dual forward" value={formatMilliseconds(forward?.dual_forward_batch)} />
      </div>
      <p className="context-note">{processing.note}</p>
    </section>
  );
}

