# Manifests

What each experiment used, so that a result can be traced to exact code, weights and data.

- `models.tsv`: source, revision and SHA-256 of the weights of every model (fetched and checked by
  `tabomics_paper.weights` for the other models, and by tabomics for TabOmics).
- `software.txt`: the machine of the runs (Python, PyTorch, CUDA, GPU, driver); the versions of every package are in
  `../requirements.lock`.
- `data.tsv`: SHA-256 of the data files (written by `scripts/01_data_manifest.py`).
- `tasks.tsv`, `folds/`: every task and its folds, with their SHA-256 (written by `scripts/02_folds.py`);
  `folds_tabpfn_wide.tsv`: whether they are those of TabPFN-Wide.

The models come from tabomics 0.3.0 (tag `v0.3.0`); the code of the experiments is this repository.
