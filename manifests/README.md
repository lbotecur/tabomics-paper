# Manifests

What each experiment used, so that a result can be traced to exact code, weights and data.

- `models.tsv`: source, revision and SHA-256 of the weights of every model.
- `software.tsv`: the versions of every package of the environment of the runs, and the GPU, driver and CUDA (to add).
- `data.tsv`: SHA-256 of the data files (written by `scripts/01_data_manifest.py`).
- `tasks.tsv`, `folds/`: every task and its folds, with their SHA-256 (written by `scripts/02_folds.py`);
  `folds_tabpfn_wide.tsv`: whether they are those of TabPFN-Wide.

The models come from tabomics 0.2.0 (tag `v0.2.0`, commit `afb29ea`); the code of the experiments is this repository.
