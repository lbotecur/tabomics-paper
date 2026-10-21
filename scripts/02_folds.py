# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""02. Every task as loaded by tabomics, and its folds (manifests/tasks.tsv, manifests/folds/).

    python scripts/02_folds.py --data-root /path/to/data            # compare with the manifests
    python scripts/02_folds.py --data-root /path/to/data --write    # write them (the runs of the paper)

For each task of HDLSS, MLOmics and Shamir: its rows, features, class counts, the SHA-256 of the prepared features
and labels, and the test rows of each fold (the training rows are the others), saved in
manifests/folds/<suite>/<task>.json with their SHA-256. Every model of the paper is evaluated on these folds, through
the loaders of tabomics. Only row indices, counts and hashes are written, never data or sample identifiers.
"""

import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from tabomics.benchmarks.runner import SUITES as MODULES

from common import MANIFESTS, SUITES, add_data_root, data_root, sha256_array, write_or_check


def fold_file(suite: str, task: str):
    """manifests/folds/<suite>/<task>.json, with "/" in the task name written as "__"."""
    return MANIFESTS / "folds" / suite / f"{task.replace('/', '__')}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--write", action="store_true", help="write the manifests instead of comparing with them")
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    args = parser.parse_args()
    root = data_root(args)
    rows, agree = [], True
    for suite in args.suites:
        module = MODULES[suite]
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
            counts = np.bincount(y)
            rows.append({"suite": suite, "task": task, "rows": len(y), "features": X.shape[1],
                         "classes": len(counts), "class_counts": " ".join(map(str, counts)), "folds": len(folds),
                         "X_sha256": sha256_array(X), "y_sha256": sha256_array(y.astype(np.int64)),
                         "folds_sha256": hashlib.sha256((content + "\n").encode()).hexdigest()})  # fmt: skip
            print(f"{suite}/{task}: {len(y)} rows, {X.shape[1]} features, {len(counts)} classes, {len(folds)} folds",
                  flush=True)  # fmt: skip
    path = MANIFESTS / ("tasks.tsv" if args.suites == list(SUITES) else f"tasks_{'_'.join(args.suites)}.tsv")
    agree &= write_or_check(pd.DataFrame(rows), path, ["suite", "task"], args.write)
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
