# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""02. Every task as loaded for the paper, and its folds (manifests/tasks.tsv, manifests/folds/).

    python scripts/02_folds.py --data-root /path/to/data            # compare with the manifests
    python scripts/02_folds.py --data-root /path/to/data --write    # write them (the runs of the paper)

For each task of HDLSS, MLOmics and Shamir, as loaded by tabomics_paper.benchmarks: its rows, features, class counts,
the SHA-256 of the prepared features and labels, and the test rows of each fold (the training rows are the others),
saved in manifests/folds/<suite>/<task>.json with their SHA-256. Every model of the paper is evaluated on these
folds, through the same loaders. Only row indices, counts and hashes are written, never data or sample identifiers.

With --tabpfn-wide (a clone of https://github.com/not-a-feature/TabPFN-Wide), the folds of HDLSS and MLOmics are also
compared with those of TabPFN-Wide: the labels of every test fold, in order, must equal the "ground_truth" of its
published per-fold results (analysis_results/wide-v2-5k; for MLOmics, which it publishes for mRNA only, the mRNA
tasks). The outcome goes to manifests/folds_tabpfn_wide.tsv. Shamir is not compared: its folds are ours.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tabomics_paper.benchmarks.runner import SUITES as MODULES

from common import MANIFESTS, SUITES, add_data_root, data_root, sha256_array, write_or_check


def fold_file(suite: str, task: str):
    """manifests/folds/<suite>/<task>.json, with "/" in the task name written as "__"."""
    return MANIFESTS / "folds" / suite / f"{task.replace('/', '__')}.json"


def published_folds(tabpfn_wide: Path, suite: str) -> pd.DataFrame | None:
    """Per-fold results published by TabPFN-Wide for its 5k model, with a "task" column named as in this repository."""
    results = tabpfn_wide / "analysis_results" / "wide-v2-5k"
    if suite == "hdlss":
        published = pd.read_csv(results / "hdlss_benchmark_results.csv")
        published["task"] = published.dataset_name
    elif suite == "mlomics":  # one row per fold and feature count; the labels are those of all the features
        published = pd.read_csv(results / "multiomics_feature_reduction_results.csv")
        largest = published.groupby(["dataset_name", "omic"]).n_features.transform("max")
        published = published[published.n_features == largest].copy()
        published["task"] = published.dataset_name + "/" + published.omic
    else:
        return None
    return published


def matches_published(published: pd.DataFrame, task: str, y: np.ndarray, folds: list) -> str:
    """Whether the labels of every test fold, in order, equal the published ones: "yes", "no" or "not published"."""
    rows = published[published.task == task].sort_values("fold")
    if rows.empty:
        return "not published"
    if len(rows) != len(folds):
        return "no"
    for (_, test), (_, row) in zip(folds, rows.iterrows(), strict=True):
        if not np.array_equal(np.array(str(row.ground_truth).split(), dtype=float).astype(int), y[test]):
            return "no"
    return "yes"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--write", action="store_true", help="write the manifests instead of comparing with them")
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--tabpfn-wide", type=Path, default=None, help="clone of the TabPFN-Wide repository")
    args = parser.parse_args()
    root = data_root(args)
    rows, agree, comparisons = [], True, []
    for suite in args.suites:
        module = MODULES[suite]
        published = published_folds(args.tabpfn_wide, suite) if args.tabpfn_wide else None
        for task in module.tasks(root):
            X, y, _ = module.load_task(task, root)
            folds = module.folds(y)
            for train, test in folds:  # the folds partition the rows, as the paper states
                assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == len(y)
            content = json.dumps({"suite": suite, "task": task, "rows": len(y),
                                  "test": [test.tolist() for _, test in folds]}, separators=(",", ":"))  # fmt: skip
            path = fold_file(suite, task)
            if args.write or not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content + "\n")
            elif path.read_text() != content + "\n":
                print(f"{suite}/{task}: the folds differ from {path.name}")
                agree = False
            if published is not None:
                comparisons.append({"suite": suite, "task": task, "same_folds_as_tabpfn_wide":
                                    matches_published(published, task, y, folds)})  # fmt: skip
            counts = np.bincount(y)
            rows.append({"suite": suite, "task": task, "rows": len(y), "features": X.shape[1],
                         "classes": len(counts), "class_counts": " ".join(map(str, counts)), "folds": len(folds),
                         "X_sha256": sha256_array(X), "y_sha256": sha256_array(y.astype(np.int64)),
                         "folds_sha256": hashlib.sha256((content + "\n").encode()).hexdigest()})  # fmt: skip
            print(f"{suite}/{task}: {len(y)} rows, {X.shape[1]} features, {len(counts)} classes, {len(folds)} folds",
                  flush=True)  # fmt: skip
    path = MANIFESTS / ("tasks.tsv" if args.suites == list(SUITES) else f"tasks_{'_'.join(args.suites)}.tsv")
    agree &= write_or_check(pd.DataFrame(rows), path, ["suite", "task"], args.write)
    if comparisons:
        comparisons = pd.DataFrame(comparisons)
        print(comparisons.same_folds_as_tabpfn_wide.value_counts().to_string())
        agree &= write_or_check(comparisons, MANIFESTS / "folds_tabpfn_wide.tsv", ["suite", "task"], args.write)
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
