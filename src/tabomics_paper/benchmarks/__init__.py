# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Benchmark suites of the paper (HDLSS, MLOmics, Shamir), with fixed folds, and a runner for any model.

Brought from tabomics 0.1.0 (tabomics.benchmarks), where they were part of the package until 0.2.0; the OpenML-CC18
suite stays out, as it is not part of the paper.
"""

from tabomics_paper.benchmarks.runner import SUITES, run_benchmark, summarize

__all__ = ["SUITES", "run_benchmark", "summarize"]
