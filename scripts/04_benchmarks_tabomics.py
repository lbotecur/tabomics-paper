# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""04. TabOmics v2 and v2.5, with 1 and 8 passes, on every fold of HDLSS, MLOmics and Shamir (results/raw/).

    python scripts/04_benchmarks_tabomics.py --data-root /path/to/data
    python scripts/04_benchmarks_tabomics.py --models tabomics-v2.5:n1      # one model only

Each model runs with the defaults of tabomics 0.3.0 (bfloat16 on CUDA, low-memory inference, the feature clip of the
released model), except n_estimators (1 or 8, the suffix ":n<k>"), random_state 42 and the categorical columns:
those of the task (none in the paper), without inference, as declared to every model. A fresh estimator is built for
each fold, on the folds of scripts/02_folds.py; the tasks in common.EXCLUDED are not run.

One CSV per model in results/raw/, one row per fold (AUROC, accuracy, seconds), written after every fold: a run
that stops resumes where it left off. Each run uses one GPU; choose it with CUDA_VISIBLE_DEVICES.

With --timing, the run measures the computational cost instead (runner.time_benchmark): the first fold of each
task, three times, after a fold that is not timed, into results/raw/timing/; on a machine running nothing else.
"""

import argparse
import sys

from tabomics import Classifier

from tabomics_paper.baselines import SEED, parse_name
from tabomics_paper.benchmarks.runner import SUITES as MODULES
from tabomics_paper.benchmarks.runner import run_benchmark, time_benchmark

from common import EXCLUDED, RESULTS, SUITES, add_data_root, data_root

WEIGHTS = {"tabomics-v2": "v2", "tabomics-v2.5": "v2.5"}  # published models of tabomics 0.3.0 (tabomics.hub)
MODELS = [f"{model}:n{passes}" for model in WEIGHTS for passes in (1, 8)]


def factory(name: str, device: str):
    """Builds the estimator of a fold of model name, given the categorical columns of the task."""
    model, n_estimators = parse_name(name)

    def make(categorical: list[int]) -> Classifier:
        return Classifier(
            model=WEIGHTS[model],
            device=device,
            categorical_features=list(categorical),
            infer_categorical=False,
            n_estimators=n_estimators,
            random_state=SEED,
        )

    return make


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retry-failed", action="store_true", help="run again the folds that failed")
    parser.add_argument("--timing", action="store_true", help="time the first fold of each task (results/raw/timing/)")
    args = parser.parse_args()
    root = data_root(args)
    tasks = [task for suite in args.suites for task in MODULES[suite].tasks(root) if task not in EXCLUDED]
    for name in args.models:
        print(f"== {name}", flush=True)
        make, csv = factory(name, args.device), f"{name.replace(':', '_')}.csv"
        if args.timing:
            time_benchmark(make, name, args.suites, root, RESULTS / "raw" / "timing" / csv, datasets=tasks)
        else:
            run_benchmark(make, name, args.suites, root, RESULTS / "raw" / csv, datasets=tasks,
                          retry_failed=args.retry_failed)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
