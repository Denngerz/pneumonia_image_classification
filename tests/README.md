# tests/ — quick automated checks (pytest)

Planned checks:
- No patient appears in more than one split (no data leakage); class ratios are preserved.
- Every registered model returns output of shape `(batch, 2)` for a `224×224` input.
- Metric function returns correct values on a small hand-made example.

Run from the repository root: `pytest`. Tests must be fast and must not need the full dataset or a GPU.
