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
python scripts/04_benchmarks_tabomics.py   # TabOmics v2 and v2.5, 1 and 8 members -> results/raw/
python scripts/05_benchmarks_baselines.py  # other foundation models (1 and 8 members), classical models -> results/raw/
python scripts/04_benchmarks_tabomics.py --development   # the same on the development datasets (appendix)
python scripts/05_benchmarks_baselines.py --development
python scripts/07_ablations.py             # TabOmics v2.5: 2 and 4 members, clip 100, fp32 -> results/raw/ablations/
python scripts/08_importance_synthetic.py  # feature recovery on synthetic tables -> results/raw/importance/
python scripts/08_importance_synthetic.py --features 10000 20000 30000  # the same, as wide as the omics tasks
python scripts/09_importance_omics.py      # gene rankings on the mRNA tasks: predictive value, stability, PAM50
python scripts/10_embeddings.py            # sample embeddings on the mRNA tasks -> results/raw/embeddings/
python scripts/10_embeddings.py --figure   # their figure -> figures/embeddings.pdf
python scripts/06_statistics.py            # last: summaries, tests, tables and figures -> results/summaries/, tables/, figures/
```

The computational cost is a separate run, `--timing` in 04, 05 and 07, on a machine that runs nothing else; the main
README gives the loop used for the paper.

Nothing these scripts write into the repository contains data or sample identifiers: only file names, counts, row
indices, summary statistics, hashes, tables and figures. The only per-sample output, the representations of the
embedding figure, stays in a cache outside the repository (`10_embeddings.py --figure`).
