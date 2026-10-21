# TabOmics paper: experiments

Code, settings, logs and per-fold results of the experiments of the TabOmics paper. The models come from
[tabomics](https://github.com/lbotecur/tabomics) 0.2.0; the benchmarks, the other models compared with TabOmics and
the analyses are in this repository.

## Reproducing the paper

### 1. Environment

Create an empty environment with Python 3.12, named for instance `tabomics-paper-env`, with conda:

```bash
conda create -n tabomics-paper-env python=3.12
conda activate tabomics-paper-env
```

or with pip:

```bash
python3.12 -m venv tabomics-paper-env
source tabomics-paper-env/bin/activate
```

Then clone this repository and install it, which installs everything the scripts need, at the versions of the paper
(`pyproject.toml`): tabomics 0.2.0 with its dependencies, PyTorch, scikit-learn, XGBoost, TabPFN-Wide, TabPFN
(TabPFN v2 and TabPFN-3.5) and TabICL.

```bash
git clone https://github.com/lbotecur/tabomics-paper
cd tabomics-paper
pip install .
```

The weights of TabPFN-3.5 are gated on Hugging Face: accept their licence on the model page and log in once with
`hf auth login`. The other weights are downloaded on first use.

### 2. Data

The data are not part of this repository: they are downloaded from their original sources, each with its own terms,
into a data folder outside the repository. Every script takes that folder with `--data-root`, or reads the environment
variable `TABOMICS_DATA`.

```bash
export TABOMICS_DATA=/path/to/data
python scripts/00_download_data.py    # HDLSS, MLOmics and Shamir
python scripts/01_data_manifest.py    # are the files those used in the paper? (SHA-256)
python scripts/02_folds.py            # are the tasks and folds those of the paper?
```

The sources, the layout of the data folder, the folds and the metrics are described in
[docs/benchmarks.md](docs/benchmarks.md). With `--tabpfn-wide <clone of https://github.com/not-a-feature/TabPFN-Wide>`,
`02_folds.py` also checks that the folds of HDLSS and MLOmics are those of TabPFN-Wide.

### 3. Experiments

Run the scripts in order; each writes its results under `results/`.

| # | Command | Experiment | Paper |
|---|---|---|---|
| 03 | `python scripts/03_audit_lgg.py` | audit of the LGG methylation task of MLOmics | Section 7.1 |
| 04 | `python scripts/04_benchmarks_tabomics.py` | TabOmics v2 and v2.5 on HDLSS, MLOmics and Shamir | Sections 7.2, 8 |
| 05 | `python scripts/05_benchmarks_baselines.py` | other TFMs and classical methods on the same folds | Sections 7.2, 8 |
| 06 | `python scripts/06_statistics.py` | Wilcoxon with Holm, Friedman with Nemenyi, critical difference diagram | Sections 7.3, 8 |
| 07 | `python scripts/07_ablations.py` | feature clip, ensemble members, low-memory path | Sections 7.4, 8 |
| 08 | `python scripts/08_importance_synthetic.py` | feature recovery on synthetic tables | Sections 7.5, 8 |
| 09 | `python scripts/09_importance_omics.py` | top-ranked genes and subtype markers | Sections 7.5, 8 |
| 10 | `python scripts/10_embeddings.py` | sample embeddings | Sections 7.6, 8 |

Scripts 04 to 10 are being written; the GPU and time each one needs will be listed here.

### 4. Results

Each table and figure of the paper will be listed here with the script and the result file it comes from.

## Layout

| Path | Contents |
|---|---|
| `src/tabomics_paper/` | the package installed by `pip install .`: benchmark suites and runner (`benchmarks/`), other models (`baselines.py`) |
| `scripts/` | one numbered script per step (00 to 10) |
| `docs/` | the benchmarks: sources, data layout, folds and metrics |
| `manifests/` | SHA-256 of the model weights and data used, and the tasks and folds |
| `configs/` | settings of each experiment |
| `logs/` | logs of the runs, and the pretraining logs of TabOmics v2 and v2.5 |
| `results/raw/` | one row per fold, as written by the runs |
| `results/summaries/` | means per task and per benchmark, and statistical tests |
| `tables/`, `figures/` | tables and figures of the paper, generated from `results/` |
| `tests/` | tests of the package (`pip install ".[dev]"`, then `pytest`) |

The benchmark suites, the runner and the definitions of the other models were part of tabomics until version 0.1.0;
they were brought here unchanged, except for their imports, when tabomics 0.2.0 kept only the models.

## Licence

GPL-3.0-or-later; see `LICENSE` and `NOTICE`.
