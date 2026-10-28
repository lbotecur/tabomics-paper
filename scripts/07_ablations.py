# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""07. Ablations of TabOmics v2.5 on every fold of HDLSS, MLOmics and Shamir (results/raw/ablations/).

    python scripts/07_ablations.py --data-root /path/to/data --models tabomics-v2.5:n1:clip100

Each ablation changes one setting of the reference model of the paper, TabOmics v2.5 with one member as run by
scripts/04_benchmarks_tabomics.py (whose results are the reference):

- tabomics-v2.5:n2, tabomics-v2.5:n4: two and four members (one and eight are in results/raw/);
- tabomics-v2.5:n1:clip100: feature clip 100, the value of pretraining, instead of the model's 10;
- tabomics-v2.5:n1:fp32: single precision instead of bfloat16 mixed precision.

Everything else as in script 04, on the same folds and tasks. One CSV per model in results/raw/ablations/, written
after every fold and resumable; with --timing, the cost protocol of script 04 into results/raw/ablations/timing/.
Each run uses one GPU; choose it with CUDA_VISIBLE_DEVICES.
"""

import argparse
import sys

from tabomics import Classifier

from tabomics_paper.baselines import SEED
from tabomics_paper.benchmarks.runner import SUITES as MODULES
from tabomics_paper.benchmarks.runner import run_benchmark, time_benchmark

from common import RESULTS, SUITES, add_data_root, data_root, tasks_to_run

# Each ablation: the Classifier settings that differ from those of script 04.
ABLATIONS = {
    "tabomics-v2.5:n2": {"n_estimators": 2},
    "tabomics-v2.5:n4": {"n_estimators": 4},
    "tabomics-v2.5:n1:clip100": {"feature_clip": 100.0},
    "tabomics-v2.5:n1:fp32": {"amp_dtype": None},
}
OUTPUT = RESULTS / "raw" / "ablations"


def factory(name: str, device: str):
    """Builds the estimator of a fold of an ablation, given the categorical columns of the task."""

    def make(categorical: list[int]) -> Classifier:
        settings = {"n_estimators": 1, **ABLATIONS[name]}
        return Classifier(model="v2.5", device=device, categorical_features=list(categorical),
                          infer_categorical=False, random_state=SEED, **settings)  # fmt: skip

    return make


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--models", nargs="+", default=list(ABLATIONS), choices=list(ABLATIONS))
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retry-failed", action="store_true", help="run again the folds that failed")
    parser.add_argument("--timing", action="store_true", help="time the first fold of each task")
    args = parser.parse_args()
    root = data_root(args)
    tasks = tasks_to_run(MODULES, args.suites, root, development=False)
    for name in args.models:
        print(f"== {name}", flush=True)
        make, csv = factory(name, args.device), f"{name.replace(':', '_')}.csv"
        if args.timing:
            time_benchmark(make, name, args.suites, root, OUTPUT / "timing" / csv, datasets=tasks)
        else:
            run_benchmark(make, name, args.suites, root, OUTPUT / csv, datasets=tasks,
                          retry_failed=args.retry_failed)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
