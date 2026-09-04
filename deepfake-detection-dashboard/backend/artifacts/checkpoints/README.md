# Checkpoints

Place `spatial_best.pth` and the official `dual_best.pth` here. They are large
trusted PyTorch artifacts and are excluded from Git.

The expected SHA-256 for `dual_best.pth` is stored in
`../model_metadata.json`. The application refuses to load a different file
while verification is enabled.
