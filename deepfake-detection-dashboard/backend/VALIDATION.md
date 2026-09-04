# Validation status

Completed before packaging:

- Compared the supplied backend with the final notebook architecture,
  preprocessing and evidence functions.
- Parsed all 25 Python files successfully with Python's abstract syntax tree.
- Parsed both JSON configuration artifacts successfully.
- Passed nine dependency-free unit tests covering triage, label thresholding,
  unsupported combined-score removal and cautious narrative wording.
- Confirmed that the active source contains no `strict=False`, softmax
  classification, Haar detector, invented combined score or obsolete model
  class name.
- Validated the ZIP archive after packaging.

Still required on the target machine because the trained weights and full ML
environment were not supplied with the four source files:

1. Install dependencies and run all tests:

   ```bash
   python -m unittest discover -s tests -v
   ```

2. Copy the two checkpoint files and download/copy the face detector assets.
3. Run `python scripts/verify_artifacts.py`.
4. Run the notebook-parity test against the preserved FF++ crops:

   ```bash
   python scripts/check_notebook_parity.py PATH_TO_FF_TEST_FAKE_CROPS
   ```

5. Start the service and confirm `GET /api/health` returns `ready: true`.
6. Test one genuine video, one manipulated video, an unreadable file and a
   video with fewer than five usable face detections before connecting React.

The parity test is the release gate. Do not treat the backend as connected to
the official model until the saved reference video matches within `1e-4`.
