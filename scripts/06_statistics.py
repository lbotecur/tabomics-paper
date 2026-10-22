# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""06. Means per task and per benchmark, Wilcoxon with Holm, Friedman with Nemenyi (results/summaries/, figures/).

    python scripts/06_statistics.py

Reads every CSV of results/raw/ (scripts 04 and 05), without the tasks in common.EXCLUDED, and follows the paper
(Section 7.3), for each metric (AUROC, accuracy) and each set of tasks (all, and each benchmark):

- per_task.csv, per_benchmark.csv: each metric averaged over the folds with valid results of a task, then over the
  tasks, every task weighing the same; with the folds that failed.
- wilcoxon.csv: each TabOmics model against each other model, Wilcoxon signed-rank test on the per-task means of the
  folds valid for both; p-values corrected with Holm over the comparisons of that TabOmics model.
- friedman.csv, ranks.csv, figures/cd_<metric>_<set>.pdf: Friedman test and Nemenyi critical difference on the models
  with valid results on every fold of every task (the folds of manifests/tasks.tsv).

A fold is valid when it succeeded (status "ok") and, for the AUROC, when it is defined (every class in the test rows).
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

from common import EXCLUDED, MANIFESTS, REPO, RESULTS, SUITES

METRICS = ("roc_auc", "accuracy")
SETS = ("all", *SUITES)
ALPHA = 0.05
FOLD = ["suite", "dataset", "fold"]


def load() -> pd.DataFrame:
    """Every fold of every model, without the excluded tasks."""
    raw = pd.concat([pd.read_csv(p) for p in sorted((RESULTS / "raw").glob("*.csv"))], ignore_index=True)
    return raw[~raw.dataset.isin(EXCLUDED)]


def folds(raw: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The valid value of a metric on every fold (rows) for every model (columns); NaN when not valid."""
    valid = raw[(raw.status == "ok") & raw[metric].notna()]
    return valid.pivot_table(index=FOLD, columns="model", values=metric)


def in_set(table: pd.DataFrame, subset: str) -> pd.DataFrame:
    """The rows of a set of tasks (all, or one benchmark), the table indexed by suite first."""
    return table if subset == "all" else table[table.index.get_level_values("suite") == subset]


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values."""
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    return np.minimum(adjusted, 1)[np.argsort(order)]


def means(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means per task over its valid folds, and per set of tasks over the tasks, with the failed folds."""
    task = ["model", "suite", "dataset"]
    valid = pd.concat([folds(raw, m).groupby(["suite", "dataset"]).mean().stack().rename(m) for m in METRICS], axis=1)
    ok = raw.assign(ok=raw.status == "ok").groupby(task).ok
    per_task = pd.DataFrame({"folds_ok": ok.sum(), "folds_failed": ok.count() - ok.sum()})
    per_task = per_task.join(valid.reorder_levels(task)).sort_index()[[*METRICS, "folds_ok", "folds_failed"]]
    rows = []
    for subset in SETS:
        table = per_task if subset == "all" else per_task.xs(subset, level="suite", drop_level=False)
        summary = table.groupby("model").agg(
            roc_auc=("roc_auc", "mean"), accuracy=("accuracy", "mean"), tasks=("accuracy", "count"),
            folds_failed=("folds_failed", "sum"))  # fmt: skip
        rows.append(summary.assign(set=subset).reset_index())
    return per_task.reset_index(), pd.concat(rows)[["model", "set", "roc_auc", "accuracy", "tasks", "folds_failed"]]


def pairwise(raw: pd.DataFrame) -> pd.DataFrame:
    """Each TabOmics model against each other model: Wilcoxon on the per-task means of the folds valid for both."""
    models = sorted(raw.model.unique())
    rows = []
    for metric in METRICS:
        table = folds(raw, metric)
        for model in (m for m in models if m.startswith("tabomics-")):
            for subset in SETS:
                group = []
                for other in (m for m in models if not m.startswith("tabomics-")):
                    both = in_set(table[[model, other]].dropna(), subset).groupby(["suite", "dataset"]).mean()
                    difference = both[model] - both[other]
                    p = wilcoxon(both[model], both[other]).pvalue if (difference != 0).any() else 1.0
                    group.append({"metric": metric, "set": subset, "model": model, "other": other,
                                  "tasks": len(both), "wins": int((difference > 0).sum()),
                                  "ties": int((difference == 0).sum()), "losses": int((difference < 0).sum()),
                                  "median_difference": difference.median(), "p": p})  # fmt: skip
                group = pd.DataFrame(group)
                group["p_holm"] = holm(group.p.to_numpy())
                group["significant"] = group.p_holm < ALPHA
                rows.append(group)
    return pd.concat(rows, ignore_index=True)


def complete_models(raw: pd.DataFrame) -> list[str]:
    """The models with a successful row on every fold of every reported task of manifests/tasks.tsv."""
    tasks = pd.read_csv(MANIFESTS / "tasks.tsv", sep="\t")
    tasks = tasks[~tasks.task.isin(EXCLUDED)]
    expected = {(s, t, f) for s, t, n in zip(tasks.suite, tasks.task, tasks.folds, strict=True) for f in range(n)}
    ok = raw[raw.status == "ok"]
    return sorted(m for m, rows in ok.groupby("model") if expected <= set(zip(*(rows[c] for c in FOLD), strict=True)))


def friedman(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Friedman test, average ranks and Nemenyi critical difference, on the complete models; one diagram each."""
    models = complete_models(raw)
    tests, ranks = [], []
    for metric in METRICS:
        task_means = folds(raw[raw.model.isin(models)], metric).groupby(["suite", "dataset"]).mean()
        for subset in SETS:
            table = in_set(task_means, subset).dropna()
            k, n = len(models), len(table)
            if k < 3 or n < 2:
                continue
            average = table.rank(axis=1, ascending=False).mean().sort_values()
            cd = studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
            test = friedmanchisquare(*table.T.to_numpy())
            tests.append({"metric": metric, "set": subset, "models": k, "tasks": n, "statistic": test.statistic,
                          "p": test.pvalue, "critical_difference": cd})  # fmt: skip
            ranks.append(average.rename("average_rank").reset_index().assign(metric=metric, set=subset))
            diagram(average, cd, REPO / "figures" / f"cd_{metric}_{subset}.pdf")
    return pd.DataFrame(tests), pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()


def diagram(average: pd.Series, cd: float, path) -> None:
    """Critical difference diagram (Demsar 2006): average ranks, and bars joining models within cd of each other."""
    k, values = len(average), average.to_numpy()
    cliques = []  # maximal groups of consecutive models within cd
    for i in range(k):
        j = max(j for j in range(i, k) if values[j] - values[i] <= cd)
        if j > i and (i == 0 or values[j] - values[i - 1] > cd):
            cliques.append((values[i], values[j]))
    top = 0.6 + 0.3 * len(cliques)  # first row of labels, below the bars
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(7, 1 + 0.3 * (len(cliques) + half)))
    ax.set_xlim(k + 0.5, 0.5)  # rank 1 (best) on the right
    ax.set_ylim(top + 0.5 * half, -1.5)
    ax.axis("off")
    ax.hlines(0, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, -0.15, 0.15, color="black", lw=1)
        ax.text(r, -0.3, str(r), ha="center", va="bottom", fontsize=8)
    ax.hlines(-1.1, 1, 1 + cd, color="black", lw=1.5)
    ax.text(1 + cd / 2, -1.2, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (model, rank) in enumerate(average.items()):  # best half on the right, the rest on the left
        right = i < half
        y = top + 0.5 * (i if right else k - 1 - i)
        x = 0.4 if right else k + 0.6
        ax.plot([rank, rank, x], [0, y, y], color="black", lw=0.8)
        ax.text(x, y, f"{model} ({rank:.2f})", ha="left" if right else "right", va="center", fontsize=8)
    for line, (start, end) in enumerate(cliques):
        ax.hlines(0.4 + 0.3 * line, start - 0.05, end + 0.05, color="black", lw=2.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    raw = load()
    output = RESULTS / "summaries"
    output.mkdir(parents=True, exist_ok=True)
    per_task, per_benchmark = means(raw)
    per_task.to_csv(output / "per_task.csv", index=False)
    per_benchmark.to_csv(output / "per_benchmark.csv", index=False)
    pairwise(raw).to_csv(output / "wilcoxon.csv", index=False)
    tests, ranks = friedman(raw)
    tests.to_csv(output / "friedman.csv", index=False)
    ranks.to_csv(output / "ranks.csv", index=False)
    print(per_benchmark[per_benchmark.set == "all"].sort_values("roc_auc", ascending=False).to_string(index=False))
    print(f"complete models (Friedman, Nemenyi): {', '.join(complete_models(raw)) or 'none'}")
    print(f"written: {output.relative_to(REPO)}/, figures/cd_*.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
