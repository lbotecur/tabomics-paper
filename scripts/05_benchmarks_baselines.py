# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""05. The other models of the paper on every fold of HDLSS, MLOmics and Shamir (results/raw/).

    python scripts/05_benchmarks_baselines.py --data-root /path/to/data --models tabpfn-3.5:n1 tabpfn-3.5:n8

The models are those of tabomics_paper.baselines, with their fixed weights (manifests/models.tsv): the other tabular
foundation models, with 1 and 8 ensemble members (the suffix ":n<k>"), and the classical models, without
hyperparameter search. A fresh estimator is built for each fold, on the folds of scripts/02_folds.py; the tasks in
common.EXCLUDED are not run.

One CSV per model in results/raw/, one row per fold (AUROC, accuracy, seconds), written after every fold: a run
that stops resumes where it left off. Each run uses one GPU; choose it with CUDA_VISIBLE_DEVICES.
"""

import argparse
import sys

from tabomics_paper.baselines import CLASSIC, FOUNDATION, make_model
from tabomics_paper.benchmarks.runner import SUITES as MODULES
from tabomics_paper.benchmarks.runner import run_benchmark

from common import EXCLUDED, RESULTS, SUITES, add_data_root, data_root

MODELS = [*CLASSIC, *(f"{model}:n{passes}" for model in FOUNDATION for passes in (1, 8))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retry-failed", action="store_true", help="run again the folds that failed")
    args = parser.parse_args()
    root = data_root(args)
    tasks = [task for suite in args.suites for task in MODULES[suite].tasks(root) if task not in EXCLUDED]
    for name in args.models:
        print(f"== {name}", flush=True)
        run_benchmark(
            lambda categorical, name=name: make_model(name, categorical, args.device),
            name,
            args.suites,
            root,
            RESULTS / "raw" / f"{name.replace(':', '_')}.csv",
            datasets=tasks,
            retry_failed=args.retry_failed,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
