# Benchmarks

The three benchmark suites of the paper, as loaded by `tabomics_paper.benchmarks` (brought from tabomics 0.1.0). Each
suite lists its tasks, loads a task from the data folder and gives its fixed folds; the runner
(`tabomics_paper.benchmarks.run_benchmark`) evaluates any scikit-learn-like estimator on those folds and writes one
row per fold to a CSV (it resumes an interrupted run).

| Suite | Tasks | Folds |
|---|---|---|
| `hdlss` | 15 high-dimensional datasets of scikit-feature [1] (500 to 22,283 features), the selection of TabPFN-Wide [2] | stratified 3-fold x 10 repeats (x 3 with >= 2500 rows), seed 42 |
| `mlomics` | 5 TCGA cancers of MLOmics [3] x (mRNA, CNV, methylation, miRNA, all four) | stratified 5-fold, seed 42 |
| `shamir` | breast, gbm and sarcoma of the TCGA benchmark of Rappoport and Shamir [4] x (mRNA, methylation, miRNA, all three) | stratified 5-fold, seed 42 |

The suites are prepared as in the code of TabPFN-Wide [5], so the folds of `hdlss` and `mlomics` are theirs
(`scripts/02_folds.py --tabpfn-wide` checks it). For `shamir`, two differences: classes with fewer patients than folds
are left out, and samples are sorted, since TabPFN-Wide's order (and so its folds) changes from run to run.

## Data

All suites read from one data folder (`--data-root`, or the environment variable `TABOMICS_DATA`), which
`scripts/00_download_data.py` fills:

```
<data_root>/
├── HDLSS/<name>.mat                                   # https://jundongl.github.io/scikit-feature/datasets.html
├── Cancer-Multi-Omics-Benchmark/Main_Dataset/...      # https://huggingface.co/datasets/AIBIC/MLOmics
└── Shamir/<cancer>/{exp,methy,mirna}, Shamir/clinical/clinical/<cancer>
                                                       # https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html
```

The data are not redistributed here: each source has its own terms.

## Metrics

AUROC (binary: of class 1; multiclass: one-vs-rest, macro average; NaN when a test fold lacks a class) and accuracy,
per fold, plus the seconds per fold, from building the model (including loading its weights) to the predicted
probabilities. Each row has a status: `ok`, `oom` (out of memory) or `error` (any other exception, with its message in
the `error` column); the metrics of a failed fold are NaN. A failed fold does not stop the run, and `retry_failed`
evaluates it again. `summarize` averages the successful folds per dataset first, so every dataset weighs the same,
and counts the failed folds.

## References

[1] J. Li, K. Cheng, S. Wang, F. Morstatter, R. P. Trevino, J. Tang and H. Liu. *Feature Selection: A Data
Perspective*. ACM Computing Surveys, 2017. [doi.org/10.1145/3136625](https://doi.org/10.1145/3136625)

[2] C. Kolberg, J. Kreuer, J. Huurdeman, S. Ouaari, K. Eggensperger and N. Pfeifer. *TabPFN-Wide: Continued
Pre-Training for Extreme Feature Counts*. arXiv:2510.06162v2, 2026 (first version 2025). [arxiv.org/abs/2510.06162v2](https://arxiv.org/abs/2510.06162v2)

[3] Z. Yang, R. Kotoge, X. Piao, Z. Chen, L. Zhu, P. Gao, Y. Matsubara, Y. Sakurai and J. Sun. *MLOmics: Cancer
Multi-Omics Database for Machine Learning*. Scientific Data 12, 913, 2025.
[doi.org/10.1038/s41597-025-05235-x](https://doi.org/10.1038/s41597-025-05235-x)

[4] N. Rappoport and R. Shamir. *Multi-omic and multi-view clustering algorithms: review and cancer benchmark*.
Nucleic Acids Research, 2018. [doi.org/10.1093/nar/gky889](https://doi.org/10.1093/nar/gky889)

[5] J. Kreuer (not-a-feature). *TabPFN-Wide*. GitHub repository.
[github.com/not-a-feature/TabPFN-Wide](https://github.com/not-a-feature/TabPFN-Wide)
