# Deepfake Evidence Dashboard

This repository contains the Flask API and React dashboard for the MSc AI
spatial-frequency deepfake detection project.

The application reproduces the final notebook protocol for a new video:

1. Sample at most 20 frames uniformly across the video.
2. Retain the highest-confidence face detected by the OpenCV ResNet-SSD model.
3. Add the validated 15% face margin, resize to 224 x 224, and reproduce the
   JPEG-quality-95 crop round trip used by the research pipeline.
4. Run the locked EfficientNet-B4 spatial model and official gated
   spatial-frequency model.
5. Average frame sigmoid scores to obtain one raw score per video.
6. Run the strict frequency-residual counterfactual using the same dual
   checkpoint.
7. Apply the FF++ validation-derived manual-review threshold.

The system never trains or selects a model. It refuses to initialise if its
required artifacts are missing or if `dual_best.pth` does not match the
recorded SHA-256 identity.

## Repository structure

```text
deepfake-detection-dashboard/
├── backend/                 Flask inference API
│   ├── app/
│   │   ├── api/             HTTP routes
│   │   ├── models/          Notebook-matched architectures and loading
│   │   └── services/        Preprocessing, inference and evidence records
│   ├── artifacts/           Metadata plus local model-file placeholders
│   ├── scripts/             Artifact and notebook-parity checks
│   └── tests/
├── frontend/                Vite + React dashboard
│   └── src/
└── docker-compose.yml       Optional container workflow
```

## Required artifacts

Copy these files from the completed Kaggle project before starting the API:

```text
backend/artifacts/checkpoints/spatial_best.pth
backend/artifacts/checkpoints/dual_best.pth
backend/artifacts/face_detector/deploy.prototxt
backend/artifacts/face_detector/res10_300x300_ssd_iter_140000.caffemodel
```

`frequency_only_best.pth` is intentionally not required. It is a separately
trained research baseline, not part of the live decision path.

The official dual checkpoint SHA-256 expected by this repository is:

```text
2499bc2020eea1f7a7f0ca448312d64c4771f81f6e834944e682c4498df8dd42
```

## Run locally

### 1. Backend

Use Python 3.11 and install a PyTorch build appropriate for the machine first.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
cp .env.example .env
python scripts/verify_artifacts.py
python run.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The API is available at `http://127.0.0.1:5000/api`.

### 2. Frontend

Use Node.js 20.19 or newer.

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. During development, Vite proxies `/api` to the
Flask server.

## API

- `GET /api/health` reports whether the locked model service is ready.
- `GET /api/model-card` returns research context, benchmarks and limitations.
- `POST /api/analyze` accepts multipart field `file` and optional Boolean field
  `include_gradcam`.

The analysis response is already an exportable JSON evidence record. It
contains raw model scores, frame evidence, the frequency counterfactual,
input-adequacy checks, timing and cautious deterministic narrative text.

## Interpretation boundaries

- Upload scores are raw model scores, not calibrated probabilities.
- Benchmark AUC values describe ranking over labelled datasets, not confidence
  in one uploaded video.
- Grad-CAM shows model sensitivity, not proof or localisation of manipulation.
- The frequency counterfactual measures the effect of removing the learned
  frequency residual from the same checkpoint. It is mechanistic evidence,
  not standalone causal proof.
- MC Dropout remains an exploratory notebook analysis and does not control the
  live dashboard decision.
- The system is a research prototype and must not be used for automatic
  accusation, legal judgement or other consequential decisions.

## Validation

Backend tests that do not require the large checkpoints can be run with:

```bash
cd backend
pytest
```

Validate the frontend with:

```bash
cd frontend
npm run build
```
