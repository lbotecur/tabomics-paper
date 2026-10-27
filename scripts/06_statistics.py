# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""06. Means, Wilcoxon with Holm, Friedman with Nemenyi, ablations, cost and the development datasets.

    python scripts/06_statistics.py

Reads the CSVs of results/raw/ (scripts 04 and 05) and follows the paper (Section 6.3), for each metric (AUROC,
accuracy) and each set of tasks (all, and each benchmark), on the tasks of the paper (without common.EXCLUDED); in
results/summaries/ and figures/:

- per_task.csv, per_benchmark.csv: each metric averaged over the folds with valid results of a task, then over the
  tasks, every task weighing the same; with the folds that failed.
- wilcoxon.csv: each TabOmics model against each other model, Wilcoxon signed-rank test on the per-task means of the
  folds valid for both; p-values corrected with Holm over the comparisons of that TabOmics model.
- friedman.csv, ranks.csv, figures/cd_<metric>_<set>.pdf: Friedman test and Nemenyi critical difference on the models
  with valid results on every fold of every task (the folds of manifests/tasks.tsv).
- ablations_means.csv, ablations_wilcoxon.csv: the ablations (results/raw/ablations/, script 07) and TabOmics v2.5 x8
  against the reference, TabOmics v2.5 x1, as above; Holm over the ablations.
- timing_per_task.csv, timing_per_benchmark.csv, figures/time_vs_features.pdf: the cost runs (--timing): per task,
  the median seconds of the repeats and the peak GPU memory; per set, the median over the tasks.
- development.csv, development_advantage.csv: the development datasets (common.DEVELOPMENT, an appendix): means per
  task, and the median difference of TabOmics v2.5 x1 with each model on them and on the tasks of the paper.
- importance_synthetic_means.csv, importance_synthetic_wilcoxon.csv (script 08): per width, the means over the
  tables, and TabOmics v2.5 x1 against each other method, Wilcoxon paired by table, Holm over the methods.
- importance_omics_*.csv (script 09): the AUROC of the top-k genes per task (the mean over folds, random genes
  averaged over draws first) and over tasks, with TabOmics v2.5 x1 against each other method, Wilcoxon paired by task,
  Holm over the methods; stability, and the PAM50 genes found, per task.
- embeddings_*.csv (script 10): the AUROC and accuracy of each representation and classifier per task (the mean
  over folds) and over tasks, with the direct predictions of the foundation models (scripts 04 and 05) on the same
  folds as a reference; TabOmics v2.5 x1 against each other representation per classifier, Wilcoxon paired by task,
  Holm over the representations.

A fold is valid when it succeeded (status "ok") and, for the AUROC, when it is defined (every class in the test rows).
Each part is skipped when its results are not there yet.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

from common import DEVELOPMENT, EXCLUDED, MANIFESTS, REPO, RESULTS, SUITES, paper_name

METRICS = ("roc_auc", "accuracy")
SETS = ("all", *SUITES)
ALPHA = 0.05
FOLD = ["suite", "dataset", "fold"]
REFERENCE = "tabomics-v2.5:n1"  # the reference of the ablations


def load(folder) -> pd.DataFrame:
    """Every row of the CSVs of a folder (not of its subfolders); empty if there are none."""
    paths = sorted(folder.glob("*.csv"))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


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
        table = per_task if subset == "all" else per_task[per_task.index.get_level_values("suite") == subset]
        summary = table.groupby("model").agg(
            roc_auc=("roc_auc", "mean"), accuracy=("accuracy", "mean"), tasks=("accuracy", "count"),
            folds_failed=("folds_failed", "sum"))  # fmt: skip
        rows.append(summary.assign(set=subset).reset_index())
    return per_task.reset_index(), pd.concat(rows)[["model", "set", "roc_auc", "accuracy", "tasks", "folds_failed"]]


def compare(table: pd.DataFrame, model: str, others: list[str], metric: str, subset: str) -> pd.DataFrame:
    """A model against each of others: Wilcoxon on the per-task means of the folds valid for both, Holm over others."""
    rows = []
    for other in others:
        both = in_set(table[[model, other]].dropna(), subset).groupby(["suite", "dataset"]).mean()
        difference = both[model] - both[other]
        p = wilcoxon(both[model], both[other]).pvalue if (difference != 0).any() else 1.0
        rows.append({"metric": metric, "set": subset, "model": model, "other": other, "tasks": len(both),
                     "wins": int((difference > 0).sum()), "ties": int((difference == 0).sum()),
                     "losses": int((difference < 0).sum()), "median_difference": difference.median(),
                     "p": p})  # fmt: skip
    group = pd.DataFrame(rows)
    group["p_holm"] = holm(group.p.to_numpy())
    group["significant"] = group.p_holm < ALPHA
    return group


def pairwise(raw: pd.DataFrame) -> pd.DataFrame:
    """Each TabOmics model against each other model."""
    models = sorted(raw.model.unique())
    others = [m for m in models if not m.startswith("tabomics-")]
    groups = []
    for metric in METRICS:
        table = folds(raw, metric)
        for model in (m for m in models if m.startswith("tabomics-")):
            groups += [compare(table, model, others, metric, subset) for subset in SETS]
    return pd.concat(groups, ignore_index=True)


def ablations(raw: pd.DataFrame, ablation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means of the ablations, and each one against the reference (the ablation as model, the reference as other)."""
    combined = pd.concat([raw[raw.model.isin([REFERENCE, "tabomics-v2.5:n8"])], ablation], ignore_index=True)
    variants = sorted(set(combined.model) - {REFERENCE})
    tests = []
    for metric in METRICS:
        table = folds(combined, metric)
        for subset in SETS:  # the reference against each ablation; signs flipped so that a gain of the ablation is >0
            group = compare(table, REFERENCE, variants, metric, subset)
            group = group.rename(columns={"model": "other", "other": "model", "wins": "losses", "losses": "wins"})
            tests.append(group.assign(median_difference=-group.median_difference))
    return means(combined)[1], pd.concat(tests, ignore_index=True)


def timing(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per task, the median seconds of the successful repeats and the peak GPU memory; per set, medians over tasks."""
    ok = runs[runs.status == "ok"]
    per_task = ok.groupby(["model", "suite", "dataset"]).agg(
        features=("features", "first"), rows=("rows", "first"), seconds=("seconds", "median"),
        peak_gpu_mb=("peak_gpu_mb", "max"), repeats=("seconds", "count")).reset_index()  # fmt: skip
    failed = runs[runs.status != "ok"].groupby("model").size()
    rows = []
    for subset in SETS:
        table = per_task if subset == "all" else per_task[per_task.suite == subset]
        summary = table.groupby("model").agg(seconds=("seconds", "median"), peak_gpu_mb=("peak_gpu_mb", "median"),
                                             tasks=("seconds", "count"))  # fmt: skip
        rows.append(summary.assign(set=subset, failed_repeats=failed.reindex(summary.index, fill_value=0)))
    return per_task, pd.concat(rows).reset_index()


def time_figure(per_task: pd.DataFrame, path) -> None:
    """Median seconds per fold against the number of features, one line per model (log-log)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, rows in per_task.sort_values("features").groupby("model"):
        ax.plot(rows.features, rows.seconds, marker="o", markersize=3, lw=0.8, label=paper_name(model))
    ax.set(xscale="log", yscale="log", xlabel="Features", ylabel="Seconds per fold (median)")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})  # the same bytes on every run
    plt.close(fig)


def development(everything: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means of every model on the development datasets, and the median difference of the reference with each model
    on them and on the tasks of the paper."""
    per_task = means(everything[everything.dataset.isin(DEVELOPMENT)])[0]
    reported = means(raw)[0]
    rows = []
    for metric in METRICS:
        for name, table in (("development", per_task), ("reported", reported)):
            wide = table.pivot_table(index="dataset", columns="model", values=metric)
            if REFERENCE not in wide:
                continue
            difference = wide.sub(wide[REFERENCE], axis=0).mul(-1).drop(columns=REFERENCE)
            rows.append(difference.median().rename("median_difference").reset_index()
                        .assign(metric=metric, tasks=name))  # fmt: skip
    advantage = pd.concat(rows).pivot_table(index=["metric", "model"], columns="tasks", values="median_difference")
    return per_task, advantage.reset_index().rename(columns={"model": "other"}).assign(model=REFERENCE)


def paired_tests(table: pd.DataFrame, model: str, family: list[str]) -> pd.DataFrame:
    """A model against each of family on the rows of table (one column per method): wins, losses, median
    difference, Wilcoxon p, and p corrected with Holm over family."""
    rows = []
    for other in family:
        both = table[[model, other]].dropna()
        difference = both[model] - both[other]
        p = wilcoxon(both[model], both[other]).pvalue if (difference != 0).any() else 1.0
        rows.append({"model": model, "other": other, "n": len(both), "wins": int((difference > 0).sum()),
                     "losses": int((difference < 0).sum()), "median_difference": difference.median(),
                     "p": p})  # fmt: skip
    rows = pd.DataFrame(rows)
    rows["p_holm"] = holm(rows.p.to_numpy())
    rows["significant"] = rows.p_holm < ALPHA
    return rows


def importance_synthetic(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Means per method and width, and TabOmics against the others per width and metric (paired by table)."""
    metrics = ["recovery_auroc", "recall_at_k", "test_auroc", "test_accuracy"]
    means_ = results.groupby(["method", "features"])[metrics].mean().reset_index()
    others = [m for m in results.method.unique() if m != REFERENCE]
    tests = []
    for features, rows in results.groupby("features"):
        for metric in metrics[:3]:
            table = rows.pivot_table(index=["informative", "seed"], columns="method", values=metric)
            tests.append(paired_tests(table, REFERENCE, others).assign(features=features, metric=metric))
    return means_.rename(columns={"method": "model"}), pd.concat(tests, ignore_index=True)


def importance_omics(predictive: pd.DataFrame, stability: pd.DataFrame, pam50: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The gene rankings: per task and over tasks, with TabOmics against the others per k (paired by task)."""
    ranked = predictive[predictive.method != "all"]
    per_fold = ranked.groupby(["task", "method", "k", "fold"])[["roc_auc", "accuracy"]].mean()  # draws averaged
    per_task = per_fold.groupby(["task", "method", "k"]).mean().reset_index()
    over_tasks = per_task.groupby(["method", "k"])[["roc_auc", "accuracy"]].mean().reset_index()
    every = predictive[predictive.method == "all"].groupby("task")[["roc_auc", "accuracy"]].mean()
    over_tasks = pd.concat([over_tasks, every.mean().to_frame().T.assign(method="all", k=0)], ignore_index=True)
    others = [m for m in per_task.method.unique() if m != REFERENCE]
    tests = []
    for k, rows in per_task.groupby("k"):
        for metric in ("roc_auc", "accuracy"):
            table = rows.pivot(index="task", columns="method", values=metric)
            tests.append(paired_tests(table, REFERENCE, others).assign(k=k, metric=metric))
    summary = pam50.groupby(["task", "method", "k"]).agg(
        hits=("hits", "mean"), expected=("expected", "mean"), markers=("markers", "first"),
        folds_significant=("p", lambda p: int((p < ALPHA).sum())), folds=("p", "size")).reset_index()  # fmt: skip
    return {
        "predictive": over_tasks.rename(columns={"method": "model"}),
        "predictive_per_task": per_task.rename(columns={"method": "model"}),
        "wilcoxon": pd.concat(tests, ignore_index=True),
        "stability": stability.groupby("method")[["jaccard_top100", "spearman"]].mean().reset_index()
        .rename(columns={"method": "model"}),
        "pam50": summary.rename(columns={"method": "model"}),
    }  # fmt: skip


def embeddings(results: pd.DataFrame, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The sample embeddings: per task and over tasks, with the direct predictions of the foundation models as the
    classifier "direct"; TabOmics against the other representations per classifier and metric (paired by task)."""
    direct = raw[raw.status == "ok"].rename(columns={"dataset": "task"}).assign(classifier="direct")
    tasks = pd.MultiIndex.from_frame(results[["suite", "task"]])
    direct = direct[direct.model.isin(results.model) & pd.MultiIndex.from_frame(direct[["suite", "task"]]).isin(tasks)]
    rows = pd.concat([results, direct[["model", "classifier", "suite", "task", "fold", *METRICS]]], ignore_index=True)
    per_task = rows.groupby(["suite", "task", "model", "classifier"])[list(METRICS)].mean().reset_index()
    over_tasks = per_task.groupby(["model", "classifier"])[list(METRICS)].mean().reset_index()
    others = [m for m in results.model.unique() if m != REFERENCE]
    tests = []
    for classifier in results.classifier.unique():
        for metric in METRICS:
            table = per_task[per_task.classifier == classifier].pivot(index=["suite", "task"], columns="model",
                                                                      values=metric)  # fmt: skip
            tests.append(paired_tests(table, REFERENCE, others).assign(classifier=classifier, metric=metric))
    return {"means": over_tasks, "per_task": per_task, "wilcoxon": pd.concat(tests, ignore_index=True)}


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
            diagram(average.rename(index=paper_name), cd, REPO / "figures" / f"cd_{metric}_{subset}.pdf")
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
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})  # the same bytes on every run
    plt.close(fig)


def named(table: pd.DataFrame) -> pd.DataFrame:
    """The table with the models (columns model and other) under their names in the paper."""
    return table.assign(**{c: table[c].map(paper_name) for c in ("model", "other") if c in table})


def main() -> int:
    everything = load(RESULTS / "raw")
    raw = everything[~everything.dataset.isin(EXCLUDED)]
    output = RESULTS / "summaries"
    output.mkdir(parents=True, exist_ok=True)
    per_task, per_benchmark = means(raw)
    named(per_task).to_csv(output / "per_task.csv", index=False)
    named(per_benchmark).to_csv(output / "per_benchmark.csv", index=False)
    named(pairwise(raw)).to_csv(output / "wilcoxon.csv", index=False)
    tests, ranks = friedman(raw)
    tests.to_csv(output / "friedman.csv", index=False)
    named(ranks).to_csv(output / "ranks.csv", index=False)
    overall = named(per_benchmark[per_benchmark.set == "all"])
    print(overall.sort_values("roc_auc", ascending=False).to_string(index=False))
    print(f"complete models (Friedman, Nemenyi): {', '.join(map(paper_name, complete_models(raw))) or 'none'}")
    ablation = load(RESULTS / "raw" / "ablations")
    if len(ablation):
        ablation = ablation[~ablation.dataset.isin(EXCLUDED)]
        ablation_means, ablation_tests = ablations(raw, ablation)
        named(ablation_means).to_csv(output / "ablations_means.csv", index=False)
        named(ablation_tests).to_csv(output / "ablations_wilcoxon.csv", index=False)
        print("ablations:", ", ".join(sorted(map(paper_name, ablation.model.unique()))))
    main_runs = load(RESULTS / "raw" / "timing")
    runs = pd.concat([main_runs, load(RESULTS / "raw" / "ablations" / "timing")])
    if len(runs):
        timing_task, timing_benchmark = timing(runs[~runs.dataset.isin(EXCLUDED)])
        named(timing_task).to_csv(output / "timing_per_task.csv", index=False)
        named(timing_benchmark).to_csv(output / "timing_per_benchmark.csv", index=False)
        main_models = timing_task.model.isin(set(main_runs.get("model", [])))  # the ablations only in the tables
        time_figure(timing_task[main_models], REPO / "figures" / "time_vs_features.pdf")
        print("timed:", ", ".join(sorted(map(paper_name, runs.model.unique()))))
    if everything.dataset.isin(DEVELOPMENT).any():
        development_task, advantage = development(everything, raw)
        named(development_task).to_csv(output / "development.csv", index=False)
        named(advantage).to_csv(output / "development_advantage.csv", index=False)
        print("development datasets:", ", ".join(sorted(development_task.dataset.unique())))
    importance = RESULTS / "raw" / "importance"
    if (importance / "synthetic.csv").exists():
        synthetic_means, synthetic_tests = importance_synthetic(pd.read_csv(importance / "synthetic.csv"))
        named(synthetic_means).to_csv(output / "importance_synthetic_means.csv", index=False)
        named(synthetic_tests).to_csv(output / "importance_synthetic_wilcoxon.csv", index=False)
        print("importance on synthetic tables: widths", ", ".join(map(str, sorted(synthetic_means.features.unique()))))
    if (importance / "omics_predictive.csv").exists():
        tables = importance_omics(*(pd.read_csv(importance / f"omics_{n}.csv") for n in ("predictive", "stability",
                                                                                          "pam50")))  # fmt: skip
        for name, table in tables.items():
            named(table).to_csv(output / f"importance_omics_{name}.csv", index=False)
        print("importance on omics: tasks", tables["predictive_per_task"].task.nunique())
    embedding_runs = sorted((RESULTS / "raw" / "embeddings").glob("*/*.csv"))
    if embedding_runs:
        tables = embeddings(pd.concat(map(pd.read_csv, embedding_runs), ignore_index=True), raw)
        for name, table in tables.items():
            named(table).to_csv(output / f"embeddings_{name}.csv", index=False)
        print("embeddings: tasks", tables["per_task"].task.nunique())
    print(f"written: {output.relative_to(REPO)}/, figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
