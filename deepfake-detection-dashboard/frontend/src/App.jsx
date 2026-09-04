import { useEffect, useState } from "react";

import UploadPanel from "./components/UploadPanel";
import ScoreComparison from "./components/ScoreComparison";
import FrameExplorer from "./components/FrameExplorer";
import ModelCard from "./components/ModelCard";
import {
  FrequencyEvidence,
  InputEvidence,
  ProcessingEvidence,
} from "./components/EvidencePanel";
import {
  ActivityIcon,
  DownloadIcon,
  InfoIcon,
  ShieldIcon,
} from "./components/Icons";
import { analyzeVideo, getHealth, getModelCard } from "./lib/api";
import { downloadJson, formatNumber, formatScore } from "./lib/format";

const STATUS_CONTENT = {
  likely_manipulated: {
    eyebrow: "Automatic screening result",
    title: "Likely manipulated",
    tone: "manipulated",
    explanation: "The dual score is above 0.5 and outside the validation-locked manual-review region.",
  },
  likely_authentic: {
    eyebrow: "Automatic screening result",
    title: "Likely authentic",
    tone: "authentic",
    explanation: "The dual score is below 0.5 and outside the validation-locked manual-review region.",
  },
  inconclusive_manual_review: {
    eyebrow: "Selective triage",
    title: "Manual review recommended",
    tone: "review",
    explanation: "The result is too close to the decision boundary for the locked automatic-coverage rule.",
  },
  unable_to_assess: {
    eyebrow: "Input adequacy",
    title: "Unable to assess",
    tone: "unable",
    explanation: "Fewer than five usable face crops were available, so no authenticity conclusion was produced.",
  },
};

function ServiceStatus({ health }) {
  const ready = health?.ready === true;
  return (
    <div className={`service-status ${ready ? "ready" : "not-ready"}`}>
      <span className="status-dot" />
      <span>{ready ? `Model ready · ${health.device}` : "Model service unavailable"}</span>
    </div>
  );
}

function ResultHeader({ result }) {
  const decision = STATUS_CONTENT[result.status] || STATUS_CONTENT.unable_to_assess;
  const score = result?.scores?.dual_score;
  const margin = result?.decision?.dual_margin;
  const threshold = result?.decision?.locked_90_coverage_margin_threshold;

  return (
    <section className={`result-banner ${decision.tone}`} aria-labelledby="result-title">
      <div className="result-symbol"><ShieldIcon size={28} /></div>
      <div className="result-copy">
        <span>{decision.eyebrow}</span>
        <h2 id="result-title">{decision.title}</h2>
        <p>{decision.explanation}</p>
      </div>
      <div className="result-numbers">
        <div><span>Dual score</span><strong>{formatScore(score)}</strong></div>
        <div><span>Boundary margin</span><strong>{formatNumber(margin, 3)}</strong></div>
        <div><span>Review threshold</span><strong>{formatNumber(threshold, 3)}</strong></div>
      </div>
    </section>
  );
}

function GradcamPanel({ result }) {
  const gradcam = result?.explainability?.gradcam;
  return (
    <section className="panel gradcam-panel" aria-labelledby="gradcam-title">
      <span className="section-kicker">Explainability</span>
      <h2 id="gradcam-title">Grad-CAM sensitivity view</h2>
      {gradcam ? (
        <div className="gradcam-content">
          <img src={gradcam.overlay_data_url} alt="Grad-CAM sensitivity overlay on a representative face crop" />
          <div>
            <p>{result.narrative?.gradcam}</p>
            <dl className="detail-list compact">
              <div><dt>Video frame</dt><dd>{gradcam.representative_frame_index}</dd></div>
              <div><dt>Frame score</dt><dd>{formatScore(gradcam.frame_score)}</dd></div>
            </dl>
            <p className="context-note">{gradcam.interpretation}</p>
          </div>
        </div>
      ) : (
        <div className="empty-inline">
          <InfoIcon />
          <p>{result.narrative?.gradcam || "Grad-CAM was not requested for this analysis."}</p>
        </div>
      )}
    </section>
  );
}

function AnalysisResult({ result }) {
  return (
    <div className="result-view">
      <div className="result-toolbar">
        <div>
          <span>Evidence record</span>
          <strong>{result.source?.filename}</strong>
        </div>
        <button className="secondary-button" onClick={() => downloadJson(result)}>
          <DownloadIcon /> Export JSON
        </button>
      </div>

      <ResultHeader result={result} />

      <div className="result-grid">
        <div className="result-main">
          <ScoreComparison result={result} />
          <FrequencyEvidence result={result} />
          <FrameExplorer frames={result.frames} />
        </div>
        <aside className="result-aside">
          <InputEvidence result={result} />
          <GradcamPanel result={result} />
          <ProcessingEvidence result={result} />
        </aside>
      </div>

      <div className="disclaimer-box">
        <InfoIcon />
        <p>{result.disclaimer}</p>
      </div>
    </div>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState("analyze");
  const [health, setHealth] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [modelCardLoading, setModelCardLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadContext() {
      const [healthResult, modelResult] = await Promise.allSettled([
        getHealth(),
        getModelCard(),
      ]);
      if (cancelled) return;
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
      else setHealth({ ready: false });
      if (modelResult.status === "fulfilled") setMetadata(modelResult.value);
      setModelCardLoading(false);
    }

    loadContext();
    return () => { cancelled = true; };
  }, []);

  async function runAnalysis(file, includeGradcam) {
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const record = await analyzeVideo(file, includeGradcam);
      setResult(record);
    } catch (analysisError) {
      setError(analysisError.message || "The video could not be analysed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Deepfake Evidence Dashboard home">
          <span className="brand-mark"><ActivityIcon /></span>
          <span><strong>Deepfake Evidence</strong><small>Spatial-frequency research dashboard</small></span>
        </a>
        <nav aria-label="Primary navigation">
          <button className={activeView === "analyze" ? "active" : ""} onClick={() => setActiveView("analyze")}>Analyse video</button>
          <button className={activeView === "model" ? "active" : ""} onClick={() => setActiveView("model")}>Model card</button>
        </nav>
        <ServiceStatus health={health} />
      </header>

      <main id="top">
        {activeView === "analyze" ? (
          <>
            <UploadPanel busy={busy} onAnalyze={runAnalysis} />
            {busy && (
              <div className="analysis-loading" role="status">
                <span className="spinner" />
                <div><strong>Analysing the sampled facial evidence</strong><span>Video decoding, face extraction and model inference may take a moment.</span></div>
              </div>
            )}
            {error && <div className="error-banner" role="alert"><InfoIcon /><div><strong>Analysis could not be completed</strong><span>{error}</span></div></div>}
            {result && <AnalysisResult result={result} />}
            {!result && !busy && !error && (
              <section className="method-strip" aria-label="Analysis stages">
                <div><span>01</span><strong>Sample</strong><small>Up to 20 frames across the video</small></div>
                <div><span>02</span><strong>Extract</strong><small>Highest-confidence face per frame</small></div>
                <div><span>03</span><strong>Compare</strong><small>Spatial and frequency-fused scores</small></div>
                <div><span>04</span><strong>Review</strong><small>Validation-locked selective triage</small></div>
              </section>
            )}
          </>
        ) : (
          <ModelCard metadata={metadata} loading={modelCardLoading} />
        )}
      </main>

      <footer>
        <p>Group 9 · MSc Artificial Intelligence · University of the West of England</p>
        <p>Research prototype · Not forensic proof</p>
      </footer>
    </div>
  );
}

