# Migration from the supplied backend

The four supplied files should not remain on the live import path.

| Supplied file | Decision | Reason |
|---|---|---|
| `models.py` | Replace | It defines torchvision two-class EfficientNet/ResNet models, not the trained timm EfficientNet-B4, FFT CNN and gated residual architecture. Its state dictionaries cannot match the real checkpoints. |
| `video_processor.py` | Replace | It imports class names that do not exist, silently loads weights with `strict=False`, classifies whole frames, uses Haar detection, softmax and an unvalidated average of spatial and dual scores. |
| `app.py` | Replace | It trusts filenames, allows unrestricted CORS, shares output filenames across requests, leaks exception text, hardcodes localhost URLs and exposes the invented combined score as the verdict. |
| `grad_cam.py` | Replace | It assumes two output classes and one tensor input, does not remove hooks, and does not reproduce the notebook’s predicted-class Grad-CAM on the dual spatial branch. |

Do not delete the teammate’s files until this replacement has been committed.
Move them to a temporary `legacy_backend/` branch or retain them in Git history,
then remove them from the active backend directory.

The corrected API intentionally removes `fftScore` and `combinedScore`.
Frequency-only AUC is a dataset-level research result, not a live score to be
averaged with the official model. The dashboard instead receives:

- the spatial comparison score;
- the official dual score;
- the score from the same dual checkpoint with its frequency residual disabled;
- the difference between the last two scores;
- the mean gate alpha, described only as a learned residual weight;
- input-adequacy warnings and a validation-locked manual-review outcome.
