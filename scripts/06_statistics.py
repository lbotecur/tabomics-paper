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
- versions.csv: TabOmics v2.5 against v2 with the same passes, as above (one comparison per family).
- tables/benchmarks_<metric>.tex: the table of the paper, the means per benchmark and over all tasks, the average
  rank, and the wins-losses and Holm p of TabOmics v2.5 x1 and x8 against each model (from wilcoxon.csv).
- friedman.csv, ranks.csv, figures/cd_<metric>_<set>.pdf: Friedman test and Nemenyi critical difference on the models
  with valid results on every fold of every task (the folds of manifests/tasks.tsv).
- ablations_means.csv, ablations_wilcoxon.csv: the ablations (results/raw/ablations/, script 07) and TabOmics v2.5 x8
  against the reference, TabOmics v2.5 x1, as above; Holm over the ablations.
- timing_per_task.csv, timing_per_benchmark.csv: the cost runs (--timing): per task,
  the median seconds of the repeats and the peak GPU memory; per set, the median over the tasks.
- tables/cost.tex: the cost table of the paper for the models of the benchmarks: the mean AUROC over all tasks, the
  median seconds per fold per benchmark and over all tasks, the slowest task, and the median and largest peak GPU
  memory (none recorded for XGBoost and the CPU methods).
- tables/ablations.tex: the ablation table of the paper: the mean AUROC and accuracy over all tasks of each ablation,
  its wins-ties-losses and Holm p against the reference, and its median seconds per fold and GPU memory.
- tables/per_task_<1,2>.tex: the AUROC of every model on every task (appendix), the TFMs that run with one and eight
  members in the first, the others in the second; the best of the 15 models on each task in bold.
- tables/development.tex: the development datasets (appendix): the AUROC of every model on each, and the median
  difference of TabOmics v2.5 x1 with each model on them and on the tasks of the paper.
- tables/top_genes.tex: the genes most often among the top 10 of TabOmics v2.5 x1 in the folds of the breast tasks,
  PAM50 genes in bold (appendix).
- development.csv, development_advantage.csv: the development datasets (common.DEVELOPMENT, an appendix): means per
  task, and the median difference of TabOmics v2.5 x1 with each model on them and on the tasks of the paper.
- importance_synthetic_means.csv, importance_synthetic_wilcoxon.csv (script 08): per width, the means over the
  tables, and TabOmics v2.5 x1 against each other method, Wilcoxon paired by table, Holm over the methods.
- importance_omics_*.csv (script 09): the AUROC of the top-k genes per task (the mean over folds, random genes
  averaged over draws first) and over tasks, with TabOmics v2.5 x1 against each other method, Wilcoxon paired by task,
  Holm over the methods; stability, and the PAM50 genes found, per task.
- figures/importance_synthetic.pdf: the recovery AUROC and the test AUROC on the synthetic tables against their
  width, mean and standard error over the 15 tables of each width.
- tables/importance_omics.tex: the gene table of the paper: the AUROC of the top-k genes over the tasks, the
  stability of the rankings, and the PAM50 genes among the top 10 of the breast tasks.
- embeddings_*.csv (script 10): the AUROC and accuracy of each representation and classifier per task (the mean
  over folds) and over tasks, with the direct predictions of the foundation models (scripts 04 and 05) on the same
  folds as a reference; TabOmics v2.5 x1 against each other representation per classifier, Wilcoxon paired by task,
  Holm over the representations.
- tables/embeddings.tex: the embedding table of the paper: per representation and classifier, the mean AUROC over
  the tasks with the wins-losses and Holm p of TabOmics v2.5 x1 against it, and the direct predictions.

A fold is valid when it succeeded (status "ok") and, for the AUROC, when it is defined (every class in the test rows).
Each part is skipped when its results are not there yet.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

from common import DEVELOPMENT, EXCLUDED, MANIFESTS, REPO, RESULTS, SUITES, paper_name
from importance import PAM50

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


def versions(raw: pd.DataFrame) -> pd.DataFrame:
    """TabOmics v2.5 against v2, with one and with eight passes."""
    tests = []
    for metric in METRICS:
        table = folds(raw, metric)
        tests += [compare(table, f"tabomics-v2.5:n{k}", [f"tabomics-v2:n{k}"], metric, subset)
                  for k in (1, 8) for subset in SETS]  # fmt: skip
    return pd.concat(tests, ignore_index=True)


def latex_name(model: str) -> str:
    return paper_name(model).replace("×", r"$\times$")


def latex_p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def benchmark_table(per_benchmark: pd.DataFrame, ranks: pd.DataFrame, tests: pd.DataFrame, metric: str) -> str:
    """The table of the paper for a metric: the mean per benchmark and over all tasks, the average rank, and the
    wins-losses and Holm p of TabOmics v2.5 x1 and x8 against each model, over all tasks; the best values in bold."""
    columns = [*SUITES, "all"]
    means_ = per_benchmark.pivot(index="model", columns="set", values=metric)[columns]
    table = means_.round(3)  # as printed: equal values are equally bold
    rank = ranks[(ranks.metric == metric) & (ranks.set == "all")].set_index("model").average_rank.round(2)
    tests = tests[(tests.metric == metric) & (tests.set == "all")].set_index(["model", "other"])
    references = ["tabomics-v2.5:n1", "tabomics-v2.5:n8"]
    lines = []
    for model in means_.sort_values("all", ascending=False).index:
        cells = [latex_name(model)]
        cells += [rf"\textbf{{{v:.3f}}}" if v == table[c].max() else f"{v:.3f}" for c, v in table.loc[model].items()]
        cells.append(rf"\textbf{{{rank[model]:.2f}}}" if rank[model] == rank.min() else f"{rank[model]:.2f}")
        for reference in references:
            test = tests.loc[(reference, model)] if (reference, model) in tests.index else None
            cells += ["--", "--"] if test is None else [f"{test.wins}--{test.losses}", latex_p(test.p_holm)]
        lines.append(" & ".join(cells) + r" \\")
    name = {"roc_auc": "AUROC", "accuracy": "accuracy"}[metric]
    header = [
        r"\begin{tabular}{lrrrrrcrcr}",
        r"\toprule",
        rf" & \multicolumn{{4}}{{c}}{{Mean {name}}} & & \multicolumn{{2}}{{c}}{{{latex_name(references[0])} vs.}} & "
        rf"\multicolumn{{2}}{{c}}{{{latex_name(references[1])} vs.}} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
        r"Model & HDLSS & MLOmics & Shamir & All & Rank & W--L & $p$ & W--L & $p$ \\",
        r"\midrule",
    ]
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


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


def cost_table(per_task: pd.DataFrame, per_set: pd.DataFrame, per_benchmark: pd.DataFrame) -> str:
    """The cost table of the paper, the models sorted by their mean AUROC over all tasks, as in the benchmark table."""
    auroc = per_benchmark[per_benchmark.set == "all"].set_index("model").roc_auc
    seconds = per_set.pivot(index="model", columns="set", values="seconds")[[*SUITES, "all"]]
    memory = per_task.groupby("model").peak_gpu_mb.agg(["median", "max"]) / 1024
    lines = []
    for model in auroc.sort_values(ascending=False).index:
        gpu = ["--", "--"] if memory.loc[model, "max"] == 0 else [f"{v:.1f}" for v in memory.loc[model]]
        cells = [latex_name(model), f"{auroc[model]:.3f}", *(f"{v:.1f}" for v in seconds.loc[model]),
                 f"{per_task[per_task.model == model].seconds.max():.0f}", *gpu]  # fmt: skip
        lines.append(" & ".join(cells) + r" \\")
    header = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{5}{c}{Seconds per fold} & \multicolumn{2}{c}{GPU memory (GB)} \\",
        r"\cmidrule(lr){3-7} \cmidrule(lr){8-9}",
        r"Model & AUROC & HDLSS & MLOmics & Shamir & All & Max & Median & Max \\",
        r"\midrule",
    ]
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


ABLATION_ROWS = {  # the rows of the ablation table, in order
    "tabomics-v2.5:n1": "One pass (reference)",
    "tabomics-v2.5:n1:clip100": "Feature clip 100",
    "tabomics-v2.5:n1:fp32": "Single precision",
    "tabomics-v2.5:n2": "Two passes",
    "tabomics-v2.5:n4": "Four passes",
    "tabomics-v2.5:n8": "Eight passes",
}


def ablation_table(means_: pd.DataFrame, tests: pd.DataFrame, per_task: pd.DataFrame, per_set: pd.DataFrame) -> str:
    """The ablation table of the paper: each metric over all tasks with the comparison with the reference, and the cost
    (-- for the clip, not timed: it does not change the computation)."""
    means_ = means_[means_.set == "all"].set_index("model")
    tests = tests[tests.set == "all"].set_index(["model", "metric"])
    seconds = per_set[per_set.set == "all"].set_index("model").seconds
    memory = per_task.groupby("model").peak_gpu_mb.median() / 1024
    lines = []
    for model, label in ABLATION_ROWS.items():
        cells = [label]
        for metric in METRICS:
            cells.append(f"{means_.loc[model, metric]:.4f}")
            if model == REFERENCE:
                cells += ["--", "--"]
            else:
                test = tests.loc[(model, metric)]
                cells += [f"{test.wins}--{test.ties}--{test.losses}", latex_p(test.p_holm)]
        cells += [f"{seconds[model]:.1f}", f"{memory[model]:.1f}"] if model in seconds.index else ["--", "--"]
        lines.append(" & ".join(cells) + r" \\")
    header = [
        r"\begin{tabular}{lrcrrcrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{AUROC} & \multicolumn{3}{c}{Accuracy} & & \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"TabOmics v2.5 & Mean & W--T--L & $p$ & Mean & W--T--L & $p$ & Seconds & GPU (GB) \\",
        r"\midrule",
    ]
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


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


IMPORTANCE_STYLES = {  # color and line of each method in the figure of the synthetic tables (TabOmics on all
    # the rows is left out: its curve lies on that of TabOmics, importance_synthetic_wilcoxon.csv)
    "tabomics-v2.5:n1": ("#1f4e79", "-"),
    "tabpfn-wide-5k:n1": ("#d95f02", "-"),
    "random_forest": ("#7f7f7f", "-"),
}


def synthetic_figure(results: pd.DataFrame, path) -> None:
    """Recovery AUROC and test AUROC against the number of features: mean and standard error over the tables."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    panels = (("recovery_auroc", "Feature recovery"), ("test_auroc", "Test rows"))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for method, (color, line) in IMPORTANCE_STYLES.items():
            values = results[results.method == method].groupby("features")[metric]
            ax.errorbar(values.mean().index, values.mean(), yerr=values.sem(), color=color, ls=line, marker="o",
                        markersize=3, lw=1.2, capsize=2, label=paper_name(method))  # fmt: skip
        ax.axhline(0.5, color="black", lw=0.6, ls=":")  # chance
        ax.axvline(5000, color="grey", lw=0.6, ls=":")  # the widest tables of TabPFN-Wide's protocol
        ax.set(xscale="log", xlabel="Features", title=title)
    axes[0].set_ylabel("AUROC")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.16))  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})  # the same bytes on every run
    plt.close(fig)


GENE_ROWS = ("tabomics-v2.5:n1", "tabpfn-wide-5k:n1", "random_forest", "anova", "random", "all")
BREAST = ("BRCA/mrna", "breast/mrna")


def importance_table(tables: dict[str, pd.DataFrame]) -> str:
    """The gene table of the paper; the best value of each column among the rankings in bold."""
    auroc = tables["predictive"].pivot(index="model", columns="k", values="roc_auc").round(3)
    ks = [k for k in auroc.columns if k > 0]
    stability = tables["stability"].set_index("model")[["jaccard_top100", "spearman"]].round(2)
    pam50 = tables["pam50"][tables["pam50"].k == 10].pivot(index="model", columns="task", values="hits")[list(BREAST)]
    pam50 = pam50.round(1)
    ranked = [m for m in GENE_ROWS if m != "all"]

    def cell(table, model, column, digits):
        if model not in table.index or pd.isna(table.loc[model, column]):
            return "--"
        value = table.loc[model, column]
        best = table.loc[table.index.intersection(ranked), column].max()
        return rf"\textbf{{{value:.{digits}f}}}" if value == best else f"{value:.{digits}f}"

    lines = []
    for model in GENE_ROWS:
        if model == "all":
            cells = [paper_name(model), rf"\multicolumn{{4}}{{c}}{{{auroc.loc[model, 0]:.3f}}}", "--", "--", "--", "--"]
        else:
            cells = [latex_name(model), *(cell(auroc, model, k, 3) for k in ks),
                     *(cell(stability, model, c, 2) for c in stability.columns),
                     *(cell(pam50, model, t, 1) for t in BREAST)]  # fmt: skip
        lines.append(" & ".join(cells) + r" \\")
    header = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{AUROC of the top $k$ genes} & \multicolumn{2}{c}{Stability} & "
        r"\multicolumn{2}{c}{PAM50 in top 10} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        "Method & " + " & ".join(f"$k={k}$" for k in ks) + r" & Jaccard & Spearman & MLOmics & Shamir \\",
        r"\midrule",
    ]
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


EMBEDDING_ROWS = ("tabomics-v2.5:n1", "tabomics-v2.5:n8", "tabpfn-wide-5k:n1", "tabpfn-v2-gn2p4bpt:n1", "all", "pls",
                  "pca")  # fmt: skip


def embeddings_table(tables: dict[str, pd.DataFrame]) -> str:
    """The embedding table of the paper, AUROC only; the best value of each column in bold."""
    auroc = tables["means"].pivot(index="model", columns="classifier", values="roc_auc").round(3)
    tests = tables["wilcoxon"][tables["wilcoxon"].metric == "roc_auc"].set_index(["classifier", "other"])
    lines = []
    for model in EMBEDDING_ROWS:
        cells = [latex_name(model)]
        for classifier in ("logreg", "knn", "direct"):
            value = auroc.loc[model, classifier]
            best = value == auroc[classifier].max()
            cells.append("--" if pd.isna(value) else rf"\textbf{{{value:.3f}}}" if best else f"{value:.3f}")
            if classifier != "direct":
                test = tests.loc[(classifier, model)] if (classifier, model) in tests.index else None
                cells += ["--", "--"] if test is None else [f"{test.wins}--{test.losses}", latex_p(test.p_holm)]
        lines.append(" & ".join(cells) + r" \\")
    header = [
        r"\begin{tabular}{lrcrrcrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Logistic regression} & \multicolumn{3}{c}{Nearest neighbors} & Direct \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8}",
        r"Representation & AUROC & W--L & $p$ & AUROC & W--L & $p$ & AUROC \\",
        r"\midrule",
    ]
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


OMIC_NAMES = {"mrna": "mRNA", "cnv": "CNV", "methylation": "methylation", "mirna": "miRNA"}


def latex_task(dataset: str) -> str:
    """The name of a task in a table: 'BRCA/mrna' -> 'BRCA mRNA', a combined task -> 'BRCA all omics'."""
    if "/" not in dataset:
        return dataset.replace("_", r"\_")
    cancer, omics = dataset.split("/")
    return f"{cancer} " + ("all omics" if "+" in omics else OMIC_NAMES[omics])


PER_TASK_GROUPS = (  # the columns of the two per-task tables: a family with its models
    (("TabOmics v2.5", ("tabomics-v2.5:n1", "tabomics-v2.5:n8")), ("TabOmics v2", ("tabomics-v2:n1", "tabomics-v2:n8")),
     ("TabPFN-3.5", ("tabpfn-3.5:n1", "tabpfn-3.5:n8")), ("TabPFN-Wide", ("tabpfn-wide-5k:n1", "tabpfn-wide-5k:n8"))),
    (("TabPFNv2", ("tabpfn-v2-gn2p4bpt:n1", "tabpfn-v2-gn2p4bpt:n8")), ("TabICLv2", ("tabicl-v2:n1", "tabicl-v2:n8")),
     ("XGBoost", ("xgboost",)), ("Random forest", ("random_forest",)), ("Elastic-net LR", ("logreg_en",))),
)  # fmt: skip


def per_task_table(per_task: pd.DataFrame, groups) -> str:
    """The AUROC per task of the models of groups, by benchmark; the best of all the models on each task in bold."""
    table = per_task.pivot_table(index=["suite", "dataset"], columns="model", values="roc_auc", sort=False).round(3)
    models = [m for _, members in groups for m in members]
    lines = []
    for suite, name in zip(SUITES, ("HDLSS", "MLOmics", "Shamir"), strict=True):
        lines.append(rf"\midrule \multicolumn{{{len(models) + 1}}}{{l}}{{\emph{{{name}}}}} \\")
        for (_, dataset), row in table.loc[[suite]].iterrows():
            cells = [rf"\textbf{{{row[m]:.3f}}}" if row[m] == row.max() else f"{row[m]:.3f}" for m in models]
            lines.append(" & ".join([latex_task(dataset), *cells]) + r" \\")
    families = " & ".join(rf"\multicolumn{{{len(members)}}}{{c}}{{{family}}}" for family, members in groups)
    rules, column = [], 2
    for _, members in groups:
        rules.append(rf"\cmidrule(lr){{{column}-{column + len(members) - 1}}}")
        column += len(members)
    passes = " & ".join(rf"$\times${m.split(':n')[1]}" if ":n" in m else "" for m in models)
    header = [rf"\begin{{tabular}}{{l{'r' * len(models)}}}", r"\toprule", f"Task & {families} \\\\", " ".join(rules),
              f" & {passes} \\\\"]  # fmt: skip
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


def development_table(per_task: pd.DataFrame, advantage: pd.DataFrame) -> str:
    """The AUROC of every model on each development dataset and their mean, with the median difference of TabOmics
    v2.5 x1 with the model on them and on the tasks of the paper; the best value of each dataset in bold."""
    table = per_task.pivot(index="model", columns="dataset", values="roc_auc")[list(DEVELOPMENT)]
    table["mean"] = table.mean(axis=1)
    table = table.sort_values("mean", ascending=False).round(3)
    difference = advantage[advantage.metric == "roc_auc"].set_index("other")[["development", "reported"]].round(3)
    lines = []
    for model, row in table.iterrows():
        cells = [rf"\textbf{{{v:.3f}}}" if v == table[c].max() else f"{v:.3f}" for c, v in row.items()]
        if model in difference.index:  # a math minus; adding 0.0 turns a rounded -0.000 into +0.000
            cells += [f"${v + 0.0:+.3f}$" for v in difference.loc[model]]
        else:
            cells += ["--", "--"]
        lines.append(" & ".join([latex_name(model), *cells]) + r" \\")
    header = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
              r" & \multicolumn{5}{c}{AUROC} & \multicolumn{2}{c}{TabOmics v2.5 $\times$1 minus model} \\",
              r"\cmidrule(lr){2-6} \cmidrule(lr){7-8}",
              r"Model & madelon & arcene & RELATHE & gisette & Mean & Development & Reported \\",
              r"\midrule"]  # fmt: skip
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


def top_genes_table(top: pd.DataFrame) -> str:
    """For each breast task, the ten genes most often among the top 10 of TabOmics v2.5 x1 over the folds (ties by
    their mean rank), with the number of folds; PAM50 genes in bold."""
    columns = []
    for task in BREAST:
        rows = top[(top.method == REFERENCE) & (top.task == task) & (top["rank"] <= 10)]
        genes = rows.groupby("gene").agg(folds=("fold", "nunique"), rank=("rank", "mean"))
        genes = genes.sort_values(["folds", "rank"], ascending=[False, True]).head(10)
        columns.append([(rf"\textbf{{{g}}}" if g in PAM50 else g, str(n)) for g, n in genes.folds.items()])
    lines = [" & ".join(a + b) + r" \\" for a, b in zip(*columns, strict=True)]
    header = [r"\begin{tabular}{lrlr}", r"\toprule",
              r"\multicolumn{2}{c}{MLOmics BRCA} & \multicolumn{2}{c}{Shamir breast} \\",
              r"\cmidrule(lr){1-2} \cmidrule(lr){3-4}", r"Gene & Folds & Gene & Folds \\", r"\midrule"]  # fmt: skip
    return "\n".join([*header, *lines, r"\bottomrule", r"\end{tabular}", ""])


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
    pairwise_tests = pairwise(raw)
    named(pairwise_tests).to_csv(output / "wilcoxon.csv", index=False)
    named(versions(raw)).to_csv(output / "versions.csv", index=False)
    tests, ranks = friedman(raw)
    tests.to_csv(output / "friedman.csv", index=False)
    named(ranks).to_csv(output / "ranks.csv", index=False)
    (REPO / "tables").mkdir(exist_ok=True)
    for number, groups in enumerate(PER_TASK_GROUPS, start=1):
        (REPO / "tables" / f"per_task_{number}.tex").write_text(per_task_table(per_task, groups))
    for metric in METRICS:
        table = benchmark_table(per_benchmark, ranks, pairwise_tests, metric)
        (REPO / "tables" / f"benchmarks_{metric}.tex").write_text(table)
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
        main_models = timing_task.model.isin(set(main_runs.get("model", [])))  # the ablations only in the CSVs
        table = cost_table(timing_task[main_models], timing_benchmark, per_benchmark)
        (REPO / "tables" / "cost.tex").write_text(table)
        if len(ablation):
            table = ablation_table(ablation_means, ablation_tests, timing_task, timing_benchmark)
            (REPO / "tables" / "ablations.tex").write_text(table)
        print("timed:", ", ".join(sorted(map(paper_name, runs.model.unique()))))
    if everything.dataset.isin(DEVELOPMENT).any():
        development_task, advantage = development(everything, raw)
        named(development_task).to_csv(output / "development.csv", index=False)
        named(advantage).to_csv(output / "development_advantage.csv", index=False)
        (REPO / "tables" / "development.tex").write_text(development_table(development_task, advantage))
        print("development datasets:", ", ".join(sorted(development_task.dataset.unique())))
    importance = RESULTS / "raw" / "importance"
    if (importance / "synthetic.csv").exists():
        synthetic_means, synthetic_tests = importance_synthetic(pd.read_csv(importance / "synthetic.csv"))
        named(synthetic_means).to_csv(output / "importance_synthetic_means.csv", index=False)
        named(synthetic_tests).to_csv(output / "importance_synthetic_wilcoxon.csv", index=False)
        synthetic_figure(pd.read_csv(importance / "synthetic.csv"), REPO / "figures" / "importance_synthetic.pdf")
        print("importance on synthetic tables: widths", ", ".join(map(str, sorted(synthetic_means.features.unique()))))
    if (importance / "omics_predictive.csv").exists():
        tables = importance_omics(*(pd.read_csv(importance / f"omics_{n}.csv") for n in ("predictive", "stability",
                                                                                          "pam50")))  # fmt: skip
        for name, table in tables.items():
            named(table).to_csv(output / f"importance_omics_{name}.csv", index=False)
        (REPO / "tables" / "importance_omics.tex").write_text(importance_table(tables))
        top = pd.read_csv(importance / "omics_top_genes.csv")
        (REPO / "tables" / "top_genes.tex").write_text(top_genes_table(top))
        print("importance on omics: tasks", tables["predictive_per_task"].task.nunique())
    embedding_runs = sorted((RESULTS / "raw" / "embeddings").glob("*/*.csv"))
    if embedding_runs:
        tables = embeddings(pd.concat(map(pd.read_csv, embedding_runs), ignore_index=True), raw)
        for name, table in tables.items():
            named(table).to_csv(output / f"embeddings_{name}.csv", index=False)
        (REPO / "tables" / "embeddings.tex").write_text(embeddings_table(tables))
        print("embeddings: tasks", tables["per_task"].task.nunique())
    print(f"written: {output.relative_to(REPO)}/, figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
