# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""09. Gene rankings on the mRNA tasks of MLOmics and Shamir (results/raw/importance/omics_*.csv).

    python scripts/09_importance_omics.py --data-root /path/to/data

The protocol of Section 6.5 of the paper, on the 8 mRNA tasks (MLOmics: BRCA, COAD, GBM, LGG, OV; Shamir: breast,
gbm, sarcoma) and their folds (scripts/02_folds.py). In each outer fold, every method ranks the genes from the outer
training rows only: TabOmics v2.5 x1 and TabPFN-Wide 5k x1 are fitted on 80% of them (context rows, stratified) and
score the other 20% (inner query rows), TabOmics reading its scores on the query rows and TabPFN-Wide on all the rows
of the same forward pass, as it does by default; the random forest and the ANOVA F statistic use all of them. The
cleaning of MLOmics (mlomics.prepare) is fitted on the outer training rows too; the genes it drops are ranked last. No
method sees the outer test rows. The columns are permuted at random in every fold, the same permutation for every
method, and the scores mapped back to the genes: every TabOmics model adds a fixed random vector to each column
position, and with the genes always at the same positions a ranking could repeat across folds through the positions.
Then:

- omics_predictive.csv: a logistic regression on standardized features, the same for every method, fitted on the k
  top-ranked genes of the outer training rows (k = 10, 50, 100, 500) and scored on the outer test rows (AUROC,
  accuracy); with random genes (five draws) and all the genes as references.
- omics_top_genes.csv: the 100 top-ranked genes of every method and fold (gene symbols).
- omics_stability.csv: per task and method, the mean over pairs of folds of the Jaccard index of the top 100 genes and
  of the Spearman correlation of the scores, both matched by gene.
- omics_pam50.csv: on the breast tasks (MLOmics BRCA, Shamir breast), whose labels are PAM50 subtypes, the PAM50 genes
  among the top 10, 50 and 100 genes, against a hypergeometric draw from the genes of the task.

A task is written when all its folds are done; a run that stops resumes at the first task not written. Only gene
symbols, counts and scores are written, never data or sample identifiers. One GPU, chosen with CUDA_VISIBLE_DEVICES.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabomics_paper.baselines import SEED
from tabomics_paper.benchmarks import mlomics, shamir
from tabomics_paper.benchmarks.metrics import roc_auc

from common import RESULTS, add_data_root, data_root
from importance import PAM50, fit_and_score, gene_symbol, top

TASKS = [("mlomics", f"{cancer}/mrna") for cancer in mlomics.CANCERS] + [
    ("shamir", f"{cancer}/mrna") for cancer in shamir.LABEL_COLUMNS
]
BREAST = ("BRCA/mrna", "breast/mrna")
METHODS = ("tabomics-v2.5:n1", "tabpfn-wide-5k:n1", "random_forest", "anova")
TOP_K = (10, 50, 100, 500)
PAM50_K = (10, 50, 100)
RANDOM_DRAWS = 5
QUERY_FRACTION = 0.2
OUTPUT = RESULTS / "raw" / "importance"


def genes(suite: str, task: str, root: Path) -> list[str]:
    """The gene symbols of the columns of an mRNA task, read as its loader reads them (same columns, same order)."""
    cancer = task.split("/")[0]
    if suite == "mlomics":
        folder = root / mlomics.FOLDER / "Main_Dataset" / "Classification_datasets" / f"GS-{cancer}" / "Original"
        columns = mlomics._load_omic(folder / f"{cancer}_{mlomics.OMIC_FILES['mrna']}.csv").columns
    else:
        columns = shamir._load_omic(root / shamir.FOLDER / cancer / shamir.OMIC_FILES["mrna"]).columns
    return [gene_symbol(c) for c in columns]


def predictive(X, y, train, test, columns) -> tuple[float, float]:
    """AUROC and accuracy on the outer test rows of a logistic regression on the given columns."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED))
    model.fit(X[train][:, columns], y[train])
    proba = model.predict_proba(X[test][:, columns])
    return roc_auc(y[test], proba), accuracy_score(y[test], proba.argmax(axis=1))


def run_task(suite: str, task: str, root: Path, device: str) -> dict[str, pd.DataFrame]:
    """Every fold of one task: the rows of the four outputs."""
    module = {"mlomics": mlomics, "shamir": shamir}[suite]
    X, y, _ = module.load_task(task, root)
    symbols = genes(suite, task, root)
    assert len(symbols) == X.shape[1], f"{task}: {len(symbols)} gene names for {X.shape[1]} columns"
    rows = {"predictive": [], "top": [], "pam50": []}
    scores = {method: [] for method in METHODS}
    for fold, (train, test) in enumerate(module.folds(y)):
        context, query = train_test_split(train, test_size=QUERY_FRACTION, stratify=y[train], random_state=SEED)
        # The cleaning of the suite fitted on the outer training rows; the columns it drops stay NaN, are never used,
        # and score NaN, below every other gene.
        X_train, X_test, kept = module.prepare(X, train, test)
        Xf = np.full(X.shape, np.nan, dtype=X.dtype)
        Xf[np.ix_(train, kept)], Xf[np.ix_(test, kept)] = X_train, X_test
        order = kept[np.random.default_rng(SEED + fold).permutation(len(kept))]  # the columns as the methods see them
        Xp = Xf[:, order]
        for method in METHODS:
            if method in ("random_forest", "anova"):  # all the outer training rows; the query rows are unused
                permuted, _ = fit_and_score(method, Xp[train], y[train], Xp[train[:1]], device=device)
            else:
                permuted, _ = fit_and_score(method, Xp[context], y[context], Xp[query], device=device)
            score = np.full(X.shape[1], np.nan)
            score[order] = permuted  # back to the genes, in the order of the task
            scores[method].append(score)
            for k in TOP_K:
                auroc, accuracy = predictive(Xf, y, train, test, top(score, k))
                rows["predictive"].append({"method": method, "fold": fold, "k": k, "draw": 0, "roc_auc": auroc,
                                           "accuracy": accuracy})  # fmt: skip
            rows["top"] += [{"method": method, "fold": fold, "rank": r + 1, "gene": symbols[c]}
                            for r, c in enumerate(top(score, 100))]  # fmt: skip
            if task in BREAST:
                universe = {symbols[c] for c in kept}
                markers = PAM50 & universe
                for k in PAM50_K:
                    hits = len({symbols[c] for c in top(score, k)} & markers)
                    p = float(hypergeom.sf(hits - 1, len(universe), len(markers), k))
                    expected = k * len(markers) / len(universe)
                    rows["pam50"].append({"method": method, "fold": fold, "k": k, "hits": hits, "expected": expected,
                                          "markers": len(markers), "p": p})  # fmt: skip
        for k in TOP_K:  # references: random genes, and all the genes
            for draw in range(RANDOM_DRAWS):
                draws = np.random.default_rng(SEED + 1000 * fold + draw)
                columns = draws.choice(kept, min(k, len(kept)), replace=False)
                auroc, accuracy = predictive(Xf, y, train, test, columns)
                rows["predictive"].append({"method": "random", "fold": fold, "k": k, "draw": draw, "roc_auc": auroc,
                                           "accuracy": accuracy})  # fmt: skip
        auroc, accuracy = predictive(Xf, y, train, test, kept)
        rows["predictive"].append({"method": "all", "fold": fold, "k": len(kept), "draw": 0, "roc_auc": auroc,
                                   "accuracy": accuracy})  # fmt: skip
        print(f"{suite}/{task} fold {fold}: done", flush=True)
    stability = []
    for method, folds in scores.items():
        pairs = list(combinations(range(len(folds)), 2))
        jaccard = [len(set(top(folds[a], 100)) & set(top(folds[b], 100))) / len(set(top(folds[a], 100))
                   | set(top(folds[b], 100))) for a, b in pairs]  # fmt: skip
        rho = [spearmanr(folds[a], folds[b], nan_policy="omit").statistic for a, b in pairs]
        stability.append({"method": method, "jaccard_top100": float(np.mean(jaccard)), "spearman": float(np.mean(rho))})
    tables = {name: pd.DataFrame(r) for name, r in rows.items()} | {"stability": pd.DataFrame(stability)}
    return {name: table.assign(suite=suite, task=task) for name, table in tables.items() if len(table)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = data_root(args)
    paths = {name: OUTPUT / f"omics_{name}.csv" for name in ("predictive", "top", "pam50", "stability")}
    paths["top"] = OUTPUT / "omics_top_genes.csv"
    done = set(pd.read_csv(paths["stability"]).task) if paths["stability"].exists() else set()
    for suite, task in TASKS:
        if task in done:
            continue
        for name, table in run_task(suite, task, root, args.device).items():
            path = paths[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():  # without rows of this task left by a run that stopped while writing it
                old = pd.read_csv(path)
                table = pd.concat([old[old.task != task], table])
            table.to_csv(path, index=False)
        print(f"{suite}/{task}: written", flush=True)
    summary = pd.read_csv(paths["predictive"]).groupby(["method", "k"]).roc_auc.mean().unstack()
    print(summary.round(3).to_string())
    print(pd.read_csv(paths["stability"]).groupby("method")[["jaccard_top100", "spearman"]].mean().round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
