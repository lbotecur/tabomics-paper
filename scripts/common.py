# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Helpers shared by the numbered scripts: the data root, the paths of this repository, SHA-256 and manifests."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "manifests"
RESULTS = REPO / "results"
DATA_ENV = "TABOMICS_DATA"
# The suites of the paper (the narrow OpenML-CC18 tables are not part of it).
SUITES = ("hdlss", "mlomics", "shamir")
# The four development datasets of HDLSS (the model selection of the paper): left out of the reported results, run
# only with --development, for an appendix.
DEVELOPMENT = ("madelon", "arcene", "RELATHE", "gisette")
# Tasks left out of every reported result, and not run by the experiments: the development datasets, and the LGG
# tasks with methylation, whose methylation profiles do not belong to their patients (scripts/03_audit_lgg.py).
EXCLUDED = (*DEVELOPMENT, "LGG/methylation", "LGG/mrna+cnv+methylation+mirna")
# The name of each model in the paper (tables, figures, manifests/models.tsv); with several passes or ensemble members,
# " ×<k>" is added: "tabomics-v2.5:n8" -> "TabOmics v2.5 ×8".
NAMES = {
    "tabomics-v2.5": "TabOmics v2.5",
    "tabomics-v2": "TabOmics v2",
    "tabpfn-3.5": "TabPFN-3.5",
    "tabpfn-v2-gn2p4bpt": "TabPFNv2",
    "tabpfn-wide-5k": "TabPFN-Wide",
    "tabicl-v2": "TabICLv2",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "logreg_en": "Elastic-net LR",
    "anova": "ANOVA F",
    "random": "Random genes",
    "all": "All genes",
    "pca": "PCA",
    "pls": "PLS",
}


# The third part of a name: an ablation of scripts/07_ablations.py, or the rows of the importance scores (scripts 08
# and 09): "tabomics-v2.5:n1:fp32" -> "TabOmics v2.5 ×1, fp32".
VARIANTS = {"clip100": "clip 100", "fp32": "fp32", "allrows": "all rows"}


def paper_name(model: str) -> str:
    """The name of a model in the paper, from its name in the code and the CSVs."""
    base, *rest = model.split(":")
    name = NAMES[base] + (f" ×{rest[0].removeprefix('n')}" if rest else "")
    return name + (f", {VARIANTS[rest[1]]}" if len(rest) > 1 else "")


def tasks_to_run(modules: dict, suites: list[str], root: Path, development: bool) -> list[str]:
    """The tasks an experiment runs: those of the paper, or with development only the development datasets."""
    tasks = [task for suite in suites for task in modules[suite].tasks(root)]
    return [t for t in tasks if t in DEVELOPMENT] if development else [t for t in tasks if t not in EXCLUDED]


def add_data_root(parser: argparse.ArgumentParser) -> None:
    """Adds --data-root, which defaults to the environment variable TABOMICS_DATA."""
    parser.add_argument(
        "--data-root",
        default=os.environ.get(DATA_ENV),
        help=f"folder with the benchmark data (default: the environment variable {DATA_ENV})",
    )


def data_root(args: argparse.Namespace) -> Path:
    """The data root given on the command line or in TABOMICS_DATA; stops if there is none."""
    if not args.data_root:
        raise SystemExit(f"Give the data folder with --data-root or the environment variable {DATA_ENV}.")
    root = Path(args.data_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a folder.")
    return root


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, as hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and values (C order)."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.dtype.str}{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_or_check(table: pd.DataFrame, path: Path, key: list[str], write: bool) -> bool:
    """Writes a manifest, or compares it with the one in the repository.

    Without write and with an existing manifest, nothing is written: the rows that differ are printed, so that a
    rerun on other data or software shows where it departs from the runs of the paper. Returns True if they agree
    (or if the manifest was written).
    """
    if write or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, sep="\t", index=False)
        print(f"written: {path.relative_to(REPO)} ({len(table)} rows)")
        return True
    expected = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    found = table.astype(str)
    merged = expected.merge(found, on=key, how="outer", suffixes=("_expected", "_found"), indicator=True)
    problems = []
    for _, row in merged.iterrows():
        name = " ".join(str(row[k]) for k in key)
        if row["_merge"] != "both":
            problems.append(f"{name}: only in the {'manifest' if row['_merge'] == 'left_only' else 'data'}")
            continue
        for column in (c for c in table.columns if c not in key):
            if row[f"{column}_expected"] != row[f"{column}_found"]:
                problems.append(f"{name}: {column} {row[f'{column}_found']} instead of {row[f'{column}_expected']}")
    if problems:
        print(f"{len(problems)} differences with {path.relative_to(REPO)}:")
        print("\n".join(f"  {p}" for p in problems))
    else:
        print(f"agrees with {path.relative_to(REPO)} ({len(table)} rows)")
    return not problems
