# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""08. Feature recovery on synthetic tables with known informative features (results/raw/importance/synthetic.csv).

    python scripts/08_importance_synthetic.py

The synthetic benchmark of TabPFN-Wide (its Appendix F): tables from scikit-learn's make_classification with 200 to
5,000 features of which 3, 5 or 7 are informative and the others noise (no redundant or repeated features), five
seeds each. Each table has 250 rows, generated together and split, stratified, into 50 training rows, as in
TabPFN-Wide, and 200 test rows that we add: the scores of TabOmics are read on the test rows, which also give the
predictive AUROC of each method. The columns are permuted at random with the seed of the table, the same permutation
for every method, since every TabOmics model adds a fixed random vector to each column position.

For each method of importance.METHODS (but anova), one row per table: the recovery AUROC (an informative column
scoring above a noise column), Recall@k, and the test AUROC and accuracy. Written after every table, resumable.
Needs no data folder; one GPU, chosen with CUDA_VISIBLE_DEVICES.
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import RESULTS
from importance import METHODS, fit_and_score, recall_at_k, recovery_auroc

FEATURES = (200, 500, 1000, 2000, 5000)
INFORMATIVE = (3, 5, 7)
SEEDS = (0, 1, 2, 3, 4)
TRAIN_ROWS, TEST_ROWS = 50, 200
OUTPUT = RESULTS / "raw" / "importance" / "synthetic.csv"


def table(features: int, informative: int, seed: int):
    """Training and test rows of one table, and the columns of the informative features."""
    X, y = make_classification(n_samples=TRAIN_ROWS + TEST_ROWS, n_features=features, n_informative=informative,
                               n_redundant=0, n_repeated=0, shuffle=False, random_state=seed)  # fmt: skip
    order = np.random.default_rng(seed).permutation(features)  # informative columns first, then permuted
    X = X[:, order]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_ROWS, stratify=y, random_state=seed)
    return X_train, X_test, y_train, y_test, np.flatnonzero(order < informative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    methods = [m for m in METHODS if m != "anova"]
    parser.add_argument("--methods", nargs="+", default=methods, choices=methods)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    results = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    done = set(map(tuple, results[["method", "features", "informative", "seed"]].values)) if len(results) else set()
    for features in FEATURES:
        for informative in INFORMATIVE:
            for seed in SEEDS:
                X_train, X_test, y_train, y_test, columns = table(features, informative, seed)
                for method in args.methods:
                    if (method, features, informative, seed) in done:
                        continue
                    start = time.time()
                    scores, proba = fit_and_score(method, X_train, y_train, X_test, device=args.device)
                    row = {"method": method, "features": features, "informative": informative, "seed": seed,
                           "recovery_auroc": recovery_auroc(scores, columns),
                           "recall_at_k": recall_at_k(scores, columns),
                           "test_auroc": roc_auc_score(y_test, proba[:, 1]),
                           "test_accuracy": accuracy_score(y_test, proba.argmax(axis=1)),
                           "seconds": round(time.time() - start, 2)}  # fmt: skip
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                    results.to_csv(OUTPUT, index=False)
                    print(f"{method} | {features} features, {informative} informative, seed {seed} | recovery AUROC "
                          f"{row['recovery_auroc']:.3f} | test AUROC {row['test_auroc']:.3f}", flush=True)  # fmt: skip
    summary = results.groupby(["method", "features"])[["recovery_auroc", "recall_at_k", "test_auroc"]].mean()
    print(summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
