# Deepfake detector Flask backend

This backend reproduces the locked inference protocol from the final research
notebook. It does not execute notebook cells and it never trains a model.

## Inference contract

1. Sample at most 20 frames uniformly over the uploaded video.
2. Run the OpenCV ResNet-SSD face detector on CPU.
3. Keep the highest-confidence face above 0.90, add a 15% margin and resize it
   to 224 × 224.
4. Apply the same JPEG-quality-95 round trip used when the research crop cache
   was created.
5. Send ImageNet-normalised RGB to the spatial branch and raw `[0,1]` RGB to
   the FFT branch.
6. Convert each frame logit with sigmoid, then average frame probabilities.
7. Use `dual_best.pth` as the official model and 0.5 as the label threshold.
8. Apply the validation-locked 90% coverage margin only to decide whether a
   case is automatically labelled or referred for manual review.

The response keeps per-video scores separate from dataset-level AUC values.
Scores are not calibrated probabilities and the result is not forensic proof.

## Directory layout

```text
deepfake-backend/
├── app/
│   ├── api/routes.py
│   ├── models/architectures.py
│   ├── models/registry.py
│   └── services/
│       ├── analysis.py
│       ├── preprocessing.py
│       ├── inference.py
│       ├── gradcam.py
│       ├── input_quality.py
│       ├── evidence.py
│       ├── narratives.py
│       └── media.py
├── artifacts/
│   ├── checkpoints/
│   ├── face_detector/
│   ├── model_metadata.json
│   └── triage_thresholds.json
├── scripts/
├── tests/
├── .env.example
├── requirements.txt
└── run.py
```

## Required private artifacts

Copy these files into the stated directories. They are intentionally excluded
from Git:

```text
artifacts/checkpoints/spatial_best.pth
artifacts/checkpoints/dual_best.pth
artifacts/face_detector/deploy.prototxt
artifacts/face_detector/res10_300x300_ssd_iter_140000.caffemodel
```

`frequency_only_best.pth` is retained with the research artifacts but is not
needed by the live backend. The counterfactual uses `dual_best.pth` with its
frequency residual disabled; it does not substitute the separately trained
frequency-only baseline.

## Setup

Use Python 3.11 or 3.12. Install the PyTorch build appropriate for the target
CPU/CUDA platform first, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# Install torch and torchvision using the official PyTorch selector.
python -m pip install -r requirements.txt
python scripts/download_face_detector.py
```

Copy `.env.example` to `.env`, then start the development service:

```bash
python run.py
```

For a single-GPU deployment, use one application worker because every worker
loads its own EfficientNet and dual checkpoint:

```bash
python -m pip install -r requirements-production.txt
gunicorn --workers 1 --threads 4 --timeout 300 --bind 0.0.0.0:5000 run:app
```

Requests are currently serialised around the shared OpenCV DNN and PyTorch
models. This is deliberate correctness protection for the dissertation
prototype, not a claim of production-scale throughput.

Endpoints:

- `GET /api/health`
- `GET /api/model-card`
- `POST /api/analyze` with multipart field `file`

The optional multipart field `include_gradcam` accepts `true` or `false`.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
```

Run `scripts/check_notebook_parity.py` against the saved FF++ crops for
`Deepfakes_004_982`. It checks the notebook reference values for the official
dual score and the frequency-disabled counterfactual.
