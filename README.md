# TabOmics paper: experiments

Code, settings, logs and per-fold results of the experiments of the TabOmics paper. The models, the benchmark data
loaders and the baselines come from tabomics 0.1.0, included as a git submodule (`tabomics/`, tag `v0.1.0`).

## Layout

| Path | Contents |
|---|---|
| `tabomics/` | tabomics at tag `v0.1.0` (git submodule) |
| `requirements/` | pinned environments: TabOmics and the classical baselines, TabPFN (TabPFN-Wide and TabPFN-3.5), TabICL |
| `manifests/` | versions, commits and SHA-256 of the code, model weights and data used |
| `configs/` | settings of each experiment |
| `scripts/` | one numbered script per experiment |
| `logs/` | logs of the runs, and the pretraining logs of TabOmics v2 and v2.5 |
| `results/raw/` | one row per fold, as written by the runs |
| `results/summaries/` | means per task and per benchmark, and statistical tests |
| `tables/`, `figures/` | tables and figures of the paper, generated from `results/` |

## Setup

```bash
git clone --recurse-submodules https://github.com/lbotecur/tabomics-paper
cd tabomics-paper
pip install -e ./tabomics
```

The baselines need their own environments (`requirements/`), since their packages pin different versions.

The data are not part of this repository. Every script takes `--data-root`, or reads the environment variable
`TABOMICS_DATA`; the expected layout of each benchmark is described in `tabomics/docs/benchmarks.md`. The SHA-256 of
the data files used are in `manifests/`.

## Experiments

| # | Script | Experiment | Paper |
|---|---|---|---|
| 01 | `scripts/01_data_manifest.py` | SHA-256 of the data files | Section 7.1 |
| 02 | `scripts/02_folds.py` | folds of every task, saved with their SHA-256 | Section 7.1 |
| 03 | `scripts/03_audit_lgg.py` | audit of the LGG methylation task of MLOmics | Section 7.1 |
| 04 | `scripts/04_benchmarks_tabomics.py` | TabOmics v2 and v2.5 on HDLSS, MLOmics and Shamir | Sections 7.2, 8 |
| 05 | `scripts/05_benchmarks_baselines.py` | other TFMs and classical methods on the same folds | Sections 7.2, 8 |
| 06 | `scripts/06_statistics.py` | Wilcoxon with Holm, Friedman with Nemenyi, critical difference diagram | Sections 7.3, 8 |
| 07 | `scripts/07_ablations.py` | feature clip, ensemble members, low-memory path | Sections 7.4, 8 |
| 08 | `scripts/08_importance_synthetic.py` | feature recovery on synthetic tables | Sections 7.5, 8 |
| 09 | `scripts/09_importance_omics.py` | top-ranked genes and subtype markers | Sections 7.5, 8 |
| 10 | `scripts/10_embeddings.py` | sample embeddings | Sections 7.6, 8 |

Each table and figure of the paper will be listed here with the script and the result file it comes from.

## Licence

GPL-3.0-or-later; see `LICENSE` and `NOTICE`.
