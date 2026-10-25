# Scripts

One numbered script per experiment; see the table in the main README. Every script takes `--data-root`, or reads the
environment variable `TABOMICS_DATA`, and runs in the environment of the main README (`pip install .`). `common.py`
holds what they share.

Scripts 01 and 02 write manifests (`manifests/`) with `--write`, as done for the runs of the paper; without it they
compare the data with the manifests and print every difference, so that a rerun shows whether it uses the same data,
tasks and folds.

```bash
export TABOMICS_DATA=/path/to/data
python scripts/00_download_data.py     # downloads the data (once)
python scripts/01_data_manifest.py     # SHA-256 of the data files
python scripts/02_folds.py             # tasks and folds (--tabpfn-wide <clone>: compare with TabPFN-Wide's)
python scripts/03_audit_lgg.py         # audit of LGG/methylation -> results/audit/lgg_methylation.json
python scripts/04_benchmarks_tabomics.py   # TabOmics v2 and v2.5, 1 and 8 passes -> results/raw/
python scripts/05_benchmarks_baselines.py  # other foundation models (1 and 8 members), classical models -> results/raw/
python scripts/06_statistics.py            # means, Wilcoxon-Holm, Friedman-Nemenyi -> results/summaries/, figures/
python scripts/07_ablations.py             # TabOmics v2.5: 2 and 4 passes, clip 100, fp32 -> results/raw/ablations/
python scripts/08_importance_synthetic.py  # feature recovery on synthetic tables -> results/raw/importance/
python scripts/08_importance_synthetic.py --features 10000 20000 30000  # the same, as wide as the omics tasks
```

Nothing these scripts write contains data or sample identifiers: only file names, counts, row indices, summary
statistics and hashes.
