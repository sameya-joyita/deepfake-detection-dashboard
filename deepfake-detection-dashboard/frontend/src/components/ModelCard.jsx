import { formatScore } from "../lib/format";

function Benchmark({ title, detail, rows }) {
  return (
    <article className="benchmark-card">
      <span>{detail}</span>
      <h3>{title}</h3>
      <dl>
        {rows.map((row) => (
          <div key={row.label}>
            <dt>{row.label}</dt>
            <dd>{row.percent ? formatScore(row.value) : row.value.toFixed(4)}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}

export default function ModelCard({ metadata, loading }) {
  if (loading) return <div className="empty-state"><span className="spinner" />Loading model card…</div>;
  if (!metadata) return <div className="empty-state">Model-card information is unavailable.</div>;

  const benchmarks = metadata.benchmarks || {};
  const ffpp = benchmarks.ffpp_source_disjoint_test || {};
  const celeb = benchmarks.celeb_df_official_test_evaluable_subset || {};
  const ablation = benchmarks.strict_frequency_residual_disabled || {};
  const frequencyOnly = benchmarks.frequency_only_baseline || {};

  return (
    <div className="model-card-page">
      <section className="model-card-intro">
        <span className="section-kicker">Model card</span>
        <h1>Scope, evidence and known limits</h1>
        <p>
          The dashboard deploys the retained gated spatial-frequency checkpoint
          from the completed source-disjoint study. The test split was not used
          to select this checkpoint.
        </p>
        <div className="model-identity">
          <div><span>Model version</span><strong>{metadata.model_version}</strong></div>
          <div><span>Official checkpoint</span><strong>{metadata.official_checkpoint}</strong></div>
          <div><span>Calibration</span><strong>{metadata.calibrated ? "Applied" : "Not applied"}</strong></div>
        </div>
      </section>

      <section className="model-section">
        <div className="section-heading">
          <span className="section-kicker">Held-out evaluation</span>
          <h2>Benchmark context</h2>
          <p>AUC describes ranking across labelled videos. It is not the confidence of a single upload.</p>
        </div>
        <div className="benchmark-grid">
          <Benchmark
            title="FaceForensics++"
            detail="Source-disjoint test · 738 videos"
            rows={[
              { label: "Spatial video AUC", value: ffpp.spatial_video_auc },
              { label: "Dual video AUC", value: ffpp.dual_video_auc },
              { label: "Evaluable coverage", value: ffpp.coverage, percent: true },
            ].filter((row) => Number.isFinite(row.value))}
          />
          <Benchmark
            title="Celeb-DF v2"
            detail="Official-list evaluable subset · zero-shot"
            rows={[
              { label: "Dual video AUC", value: celeb.dual_video_auc },
              { label: "Evaluable coverage", value: celeb.coverage, percent: true },
            ].filter((row) => Number.isFinite(row.value))}
          />
          <Benchmark
            title="Frequency evidence"
            detail="Separate mechanistic and baseline checks"
            rows={[
              { label: "Residual-disabled AUC", value: ablation.video_auc },
              { label: "Frequency-only AUC", value: frequencyOnly.video_auc },
            ].filter((row) => Number.isFinite(row.value))}
          />
        </div>
      </section>

      <section className="model-section two-column-section">
        <div>
          <span className="section-kicker">Intended use</span>
          <h2>Analyst decision support</h2>
          <p>
            Educational and research screening of videos containing face-swap or
            facial-reenactment manipulation, with manual review for inconclusive
            cases. It is not an identity-verification or forensic-proof system.
          </p>
        </div>
        <div>
          <span className="section-kicker">Known limitations</span>
          <h2>What has not been established</h2>
          <ul className="limitation-list">
            {(metadata.limitations || []).map((limit) => <li key={limit}>{limit}</li>)}
          </ul>
        </div>
      </section>
    </div>
  );
}
