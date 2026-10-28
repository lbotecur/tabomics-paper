# TabOmics paper: experiments

Code, settings, logs and per-fold results of the experiments of the TabOmics paper. The models come from
[tabomics](https://github.com/lbotecur/tabomics) 0.3.0; the benchmarks, the other models compared with TabOmics and
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
(`pyproject.toml`): tabomics 0.3.0 with its dependencies, PyTorch, scikit-learn, XGBoost, TabPFN-Wide, TabPFN
(TabPFN v2 and TabPFN-3.5) and TabICL.

```bash
git clone https://github.com/lbotecur/tabomics-paper
cd tabomics-paper
pip install .
```

For exactly the environment of the runs of the paper, every package at the version it had, install the versions
of `requirements.lock` first and then this repository without letting pip change them:

```bash
pip install -r requirements.lock
pip install . --no-deps
```

Each run uses one GPU. The runs of the paper used one NVIDIA A100 with 80 GB (`manifests/software.txt` records the
machine); a GPU with less memory may not fit the widest tables.

The weights of every model are downloaded on first use, each at a fixed revision and checked against its SHA-256
(`manifests/models.tsv`). Those of TabPFN-3.5 are gated on Hugging Face: accept their licence on the model page and
log in once with `hf auth login`.

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

Every script writes its results under `results/`, after every fold, so that a run that stops resumes where it left
off (with `--retry-failed`, 04 and 05 also run again the folds that failed). Each run uses one GPU, chosen with
`CUDA_VISIBLE_DEVICES`. The runs of the paper started one process per model, each with its own log, as in:

```bash
export TABOMICS_DATA=/path/to/data
CUDA_VISIBLE_DEVICES=0 nohup python scripts/04_benchmarks_tabomics.py --models tabomics-v2.5:n1 \
    > logs/04_tabomics-v2.5_n1.log 2>&1 &
```

| # | Script | Experiment | Results | Logs | Paper |
|---|---|---|---|---|---|
| 03 | `03_audit_lgg.py` | audit of the LGG methylation tasks of MLOmics | `results/audit/` | `logs/03_audit_lgg.log` | Section 6.1 |
| 04 | `04_benchmarks_tabomics.py` | TabOmics v2 and v2.5, 1 and 8 passes | `results/raw/` | `logs/04_<model>.log` | Sections 6.2, 7 |
| 05 | `05_benchmarks_baselines.py` | TabPFN-3.5, TabPFNv2, TabPFN-Wide and TabICLv2 (1 and 8 members), random forest, XGBoost, elastic-net LR | `results/raw/` | `logs/05_<model>.log` | Sections 6.2, 7 |
| 04, 05 | the same, with `--development` | the development datasets | `results/raw/` | `logs/development/` | Appendix |
| 07 | `07_ablations.py` | TabOmics v2.5: 2 and 4 passes, feature clip 100, fp32 | `results/raw/ablations/` | `logs/ablations/` | Sections 6.4, 7 |
| 08 | `08_importance_synthetic.py`, then with `--features 10000 20000 30000` | feature recovery on synthetic tables | `results/raw/importance/` | `logs/importance/` | Sections 6.5, 7 |
| 09 | `09_importance_omics.py` | gene rankings on the mRNA tasks | `results/raw/importance/` | `logs/importance/` | Sections 6.5, 7 |
| 10 | `10_embeddings.py`, then with `--figure` | sample embeddings on the mRNA tasks, and their figure | `results/raw/embeddings/` | `logs/embeddings/10_<model>.log` | Sections 6.6, 7 |
| 06 | `06_statistics.py`, after the others | means, Wilcoxon with Holm, Friedman with Nemenyi, figures | `results/summaries/`, `figures/` | | Sections 6.3, 7 |

`<model>` is the name of a model with `:` written as `_` (`tabomics-v2.5:n1` → `tabomics-v2.5_n1`); `--help` lists
the models of each script. 08 and 09 run their methods in one process, 03 and 06 need no GPU. The development datasets
were run as the other tasks, with `--development` added, and the second run of 08 with `--features 10000 20000 30000`:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/04_benchmarks_tabomics.py --development --models tabomics-v2.5:n1 \
    > logs/development/04_tabomics-v2.5_n1.log 2>&1 &
```

**Figure of the embeddings.** `python scripts/10_embeddings.py --figure` draws `figures/embeddings.pdf` from the first
fold of Shamir breast, after the runs of 10. The first time it computes the five representations of the figure in one
process (about 40 minutes on one GPU) and keeps them in `~/.cache/tabomics-paper/embeddings_figure.npz`, outside the
repository, since they are per sample; later runs only draw, in seconds. Delete that file to compute them again.

**Computational cost.** The timings are a separate run, `--timing` in 04, 05 and 07: the first fold of each task,
three times after a fold that is not timed, into `results/raw/timing/` and `results/raw/ablations/timing/`. They need
a machine that runs nothing else; the paper ran every model one after another on the same GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash -c '
for m in tabomics-v2:n1 tabomics-v2:n8 tabomics-v2.5:n1 tabomics-v2.5:n8; do
    python scripts/04_benchmarks_tabomics.py --timing --models $m > logs/timing/04_${m//:/_}.log 2>&1; done
for m in random_forest logreg_en xgboost tabpfn-3.5:n1 tabpfn-3.5:n8 tabpfn-v2-gn2p4bpt:n1 tabpfn-v2-gn2p4bpt:n8 \
         tabpfn-wide-5k:n1 tabpfn-wide-5k:n8 tabicl-v2:n1 tabicl-v2:n8; do
    python scripts/05_benchmarks_baselines.py --timing --models $m > logs/timing/05_${m//:/_}.log 2>&1; done
for m in tabomics-v2.5:n2 tabomics-v2.5:n4 tabomics-v2.5:n1:fp32; do
    python scripts/07_ablations.py --timing --models $m > logs/timing/07_${m//:/_}.log 2>&1; done' &
```

**Machines.** The runs of the paper used two identical machines, each with eight NVIDIA A100 80 GB GPUs and the same
environment (`manifests/software.txt`): the benchmarks (04, 05) and every timing on the first, the timings on its GPU 0
alone; the ablations (07), the development datasets, 08, 09 and 10 on the second, and the figure of 10 on the first.
Summed over their folds, the
benchmark runs took 32 GPU hours for 04 (3 to 13 per model) and about 97 for 05 (1 to 26 per model), and 07 took 27.

### 4. Results

`06_statistics.py` writes, from the per-fold results of `results/raw/`, the summaries and tests into
`results/summaries/`, the tables of the paper into `tables/` (as LaTeX, so that no number is copied by hand) and its
figures into `figures/`:

| Paper | Summaries (`results/summaries/`) | Tables (`tables/`) | Figures (`figures/`) |
|---|---|---|---|
| 7.1 Predictive performance | `per_benchmark.csv`, `per_task.csv`, `wilcoxon.csv`, `friedman.csv`, `ranks.csv`, `versions.csv` | `benchmarks_roc_auc.tex`, `benchmarks_accuracy.tex` | `cd_roc_auc_all.pdf` (and per benchmark and metric) |
| 7.2 Computational cost | `timing_per_benchmark.csv`, `timing_per_task.csv` | `cost.tex` | |
| 7.3 Ablations | `ablations_means.csv`, `ablations_wilcoxon.csv` | `ablations.tex` | |
| 7.4 Feature importance | `importance_synthetic_*.csv`, `importance_omics_*.csv`; the top genes of every fold in `results/raw/importance/omics_top_genes.csv` | `importance_omics.tex` | `importance_synthetic.pdf` |
| 7.5 Sample embeddings | `embeddings_means.csv`, `embeddings_per_task.csv`, `embeddings_wilcoxon.csv` | `embeddings.tex` | `embeddings.pdf` (by 10 with `--figure`) |
| Appendix A, per-task results | `per_task.csv` | `per_task_1.tex`, `per_task_2.tex` | |
| Appendix B, development datasets | `development.csv`, `development_advantage.csv` | `development.tex` | |
| Appendix C, top-ranked genes | `results/raw/importance/omics_top_genes.csv` | `top_genes.tex` | |
| 6.1, audit of LGG methylation | `results/audit/lgg_methylation.json` (by 03) | | |

Rerunning 06 on the results of this repository, in the environment of `requirements.lock`, gives the same files, byte
for byte.

## Layout

| Path | Contents |
|---|---|
| `src/tabomics_paper/` | the package installed by `pip install .`: benchmark suites and runner (`benchmarks/`), other models (`baselines.py`) |
| `scripts/` | one numbered script per step (00 to 10) |
| `docs/` | the benchmarks: sources, data layout, folds and metrics |
| `manifests/` | SHA-256 of the model weights and data used, and the tasks and folds |
| `logs/` | logs of the runs, one per script and model |
| `results/raw/` | one row per fold, as written by the runs |
| `results/summaries/` | means per task and per benchmark, and statistical tests |
| `results/audit/` | the audit of the LGG methylation tasks (script 03) |
| `tables/`, `figures/` | tables and figures of the paper, generated from `results/` (scripts 06 and 10) |
| `tests/` | tests of the package (`pip install ".[dev]"`, then `pytest`) |

The benchmark suites, the runner and the definitions of the other models were part of tabomics until version 0.1.0;
they were brought here unchanged, except for their imports, when tabomics 0.2.0 kept only the models.

## Licence

GPL-3.0-or-later; see `LICENSE` and `NOTICE`.
