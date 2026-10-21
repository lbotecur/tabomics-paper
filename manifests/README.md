# Manifests

What each experiment used, so that a result can be traced to exact code, weights and data.

- `models.tsv`: source, revision and SHA-256 of the weights of every model.
- `software.tsv`: package versions of each environment (to add).
- `data.tsv`: SHA-256 of the data files (written by `scripts/01_data_manifest.py`).
- `folds/`: the folds of every task with their SHA-256 (written by `scripts/02_folds.py`).

The code of TabOmics is tabomics 0.1.0, tag `v0.1.0`, commit `3494338`.
