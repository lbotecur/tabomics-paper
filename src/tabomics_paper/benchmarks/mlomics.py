# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""MLOmics suite: TCGA cancer-subtype classification (BRCA, COAD, GBM, LGG, OV), as in TabPFN-Wide.

Read as TabPFN-Wide reads them (analysis/load_mm_data.py and analysis/data.py), so the tasks and folds are theirs:
duplicated feature names dropped, stratified 5-fold, seed 42. TabPFN-Wide also drops the features that are zero in
every sample and fills missing values with the feature mean, both over all the samples; here both are fitted on the
training rows of each split instead (prepare), so that no value of the test rows is used.
A task is a cancer and one omic or several concatenated: "BRCA/mrna", "BRCA/mrna+cnv+methylation+mirna".

Data: Main_Dataset of MLOmics (https://huggingface.co/datasets/AIBIC/MLOmics; scripts/00_download_data.py), at
<data_root>/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/GS-<CANCER>/Original/.

Brought from tabomics 0.1.0 (src/tabomics/benchmarks/mlomics.py), where it was part of the package until 0.2.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

FOLDER = "Cancer-Multi-Omics-Benchmark"
CANCERS = ("BRCA", "COAD", "GBM", "LGG", "OV")
OMIC_FILES = {"mrna": "mRNA", "cnv": "CNV", "methylation": "Methy", "mirna": "miRNA"}
ALL_OMICS = "+".join(OMIC_FILES)
N_SPLITS, SEED = 5, 42


def tasks(data_root: str | Path) -> list[str]:
    """Each cancer with each omic alone and with all four together."""
    return [f"{cancer}/{omics}" for cancer in CANCERS for omics in (*OMIC_FILES, ALL_OMICS)]


def _load_omic(path: Path) -> pd.DataFrame:
    """One omic file (features x samples) -> samples x features, duplicated feature names dropped."""
    data = pd.read_csv(path, index_col=0)
    data = data[~data.index.duplicated(keep="first")].T
    data.columns = data.columns.astype(str)
    return data


def load_task(name: str, data_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Features (float32, missing values as NaN; see prepare), labels 0..C-1, and the categorical columns (none:
    copy-number levels are ordinal)."""
    cancer, _, omics = name.partition("/")
    if cancer not in CANCERS:
        raise ValueError(f"{cancer} is not one of {CANCERS}")
    folder = Path(data_root) / FOLDER / "Main_Dataset" / "Classification_datasets" / f"GS-{cancer}" / "Original"
    parts = [_load_omic(folder / f"{cancer}_{OMIC_FILES[o]}.csv") for o in (omics or "mrna").split("+")]
    labels = pd.read_csv(folder / f"{cancer}_label_num.csv").values.flatten()  # same row order as the omics
    X = pd.concat(parts, axis=1)
    return X.to_numpy(dtype=np.float32), LabelEncoder().fit_transform(labels), []


def folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified 5-fold, one repeat, seed 42."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=1, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y))


def prepare(X: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cleaning of TabPFN-Wide fitted on the training rows: the features that are zero in every training row are
    dropped, and missing values are filled with the mean of the training rows (0 if a feature has none).

    Returns:
        The training rows, the test rows, and the indices of the columns kept.
    """
    fit = X[train]
    columns = np.flatnonzero(~(fit == 0).all(axis=0))
    with np.errstate(all="ignore"):
        means = np.nan_to_num(np.nanmean(fit[:, columns], axis=0, dtype=np.float64)).astype(X.dtype)

    def fill(rows: np.ndarray) -> np.ndarray:
        part = X[rows][:, columns]
        return np.where(np.isnan(part), means, part)

    return fill(train), fill(test), columns
