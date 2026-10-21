# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""03. Audit of the LGG methylation task of MLOmics (results/audit/lgg_methylation.json).

    python scripts/03_audit_lgg.py --data-root /path/to/data

Every model performs at chance level on LGG/methylation, although methylation is known to separate LGG subtypes.
The loader of tabomics, as that of TabPFN-Wide, takes the labels by position, in the order of the label file, for
every omic. This script checks, without changing anything:

1. the sample identifiers of each omic file of every cancer: duplicates, and whether their set and order agree with
   those of mRNA; and whether the rows of the combined task follow the order of mRNA;
2. the label file of every cancer: its rows and columns, and whether it carries sample identifiers;
3. the methylation features of every cancer: missing values, constant, all-zero and duplicated features, range;
4. the predictability of LGG: AUROC of a logistic regression on the folds of the task, for mRNA (reference),
   methylation as loaded, methylation with its samples put in the order of mRNA (when the identifiers allow it),
   and methylation with permuted labels (chance).

Only counts and summary statistics are written, never data or sample identifiers.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tabomics.benchmarks import mlomics
from tabomics.benchmarks.metrics import roc_auc

from common import RESULTS, add_data_root, data_root

CANCER, OMIC = "LGG", "methylation"
TOP_VARIANCE = 2000  # features kept, by variance on the training rows, for the logistic regression
SEED = 0


def folder(root: Path, cancer: str) -> Path:
    """The folder of a cancer, as the loader of tabomics reads it."""
    return root / mlomics.FOLDER / "Main_Dataset" / "Classification_datasets" / f"GS-{cancer}" / "Original"


def patient(sample: str) -> str:
    """Sample identifier -> patient barcode ('tcga.ab.1234.01' -> 'TCGA-AB-1234'), to compare files that write
    identifiers differently."""
    return str(sample).replace(".", "-").upper()[:12]


def raw_omic(root: Path, cancer: str, omic: str) -> pd.DataFrame:
    """An omic file as stored: features x samples."""
    return pd.read_csv(folder(root, cancer) / f"{cancer}_{mlomics.OMIC_FILES[omic]}.csv", index_col=0)


def samples_report(tables: dict[str, pd.DataFrame]) -> dict:
    """Identifiers of each omic file compared with those of mRNA, raw and as patient barcodes."""
    reference = list(tables["mrna"].columns)
    reference_patients = [patient(s) for s in reference]
    report = {}
    for omic, table in tables.items():
        ids = list(table.columns)
        patients = [patient(s) for s in ids]
        same_length = len(ids) == len(reference)
        differing = sum(a != b for a, b in zip(patients, reference_patients, strict=True)) if same_length else None
        report[omic] = {
            "samples": len(ids),
            "duplicated_identifiers": len(ids) - len(set(ids)),
            "duplicated_patients": len(patients) - len(set(patients)),
            "identifiers_same_set_as_mrna": set(ids) == set(reference),
            "patients_same_set_as_mrna": set(patients) == set(reference_patients),
            "patients_in_common_with_mrna": len(set(patients) & set(reference_patients)),
            "same_order_as_mrna": patients == reference_patients,
            "positions_differing_from_mrna": differing,
            "sorted_by_identifier": ids == sorted(ids),
        }
    return report


def combined_follows_mrna(tables: dict[str, pd.DataFrame]) -> bool:
    """Whether the rows of the combined task, built as the loader builds it, are in the order of mRNA."""
    parts = []
    for table in tables.values():
        data = table.loc[~(table == 0).all(axis=1)]
        parts.append(data[~data.index.duplicated(keep="first")].T)
    return list(pd.concat(parts, axis=1).index) == list(tables["mrna"].columns)


def labels_report(root: Path, cancer: str, samples: int) -> dict:
    """The label file: shape, and whether its header or values look like sample identifiers."""
    labels = pd.read_csv(folder(root, cancer) / f"{cancer}_label_num.csv")
    text = labels.astype(str)
    return {
        "rows": len(labels),
        "columns": labels.shape[1],
        "rows_equal_samples_of_mrna": len(labels) == samples,
        "header_looks_like_identifier": any(str(c).upper().startswith("TCGA") for c in labels.columns),
        "values_look_like_identifiers": bool(text.apply(lambda c: c.str.upper().str.startswith("TCGA")).any().any()),
        "class_counts": labels.iloc[:, -1].value_counts().sort_index().astype(int).tolist(),
        "other_label_files": sorted(p.name for p in folder(root, cancer).glob("*label*")),
    }


def features_report(table: pd.DataFrame) -> dict:
    """Missing values, constant, all-zero and duplicated features, range and sample similarity of an omic file."""
    values = table.to_numpy(dtype=float)
    std = np.nanstd(values, axis=1)
    top = values[np.argsort(-np.nan_to_num(std))[:TOP_VARIANCE]]
    correlation = pd.DataFrame(top).corr().to_numpy()
    return {
        "features": int(values.shape[0]),
        "missing_fraction": float(np.isnan(values).mean()),
        "all_zero_features": int((np.nan_to_num(values) == 0).all(axis=1).sum()),
        "constant_features": int((std == 0).sum()),
        "duplicated_feature_names": int(table.index.duplicated().sum()),
        "min": float(np.nanmin(values)),
        "median": float(np.nanmedian(values)),
        "max": float(np.nanmax(values)),
        "median_correlation_between_samples": float(np.nanmedian(correlation[np.triu_indices_from(correlation, 1)])),
    }


def cross_validated_auroc(X: np.ndarray, y: np.ndarray) -> float:
    """Mean AUROC over the folds of the task of a logistic regression on the most variable features."""
    scores = []
    for train, test in mlomics.folds(y):
        keep = np.argsort(-X[train].var(axis=0))[:TOP_VARIANCE]
        scaler = StandardScaler().fit(X[train][:, keep])
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(X[train][:, keep]), y[train])
        scores.append(roc_auc(y[test], model.predict_proba(scaler.transform(X[test][:, keep]))))
    return float(np.nanmean(scores))


def predictability(root: Path, tables: dict[str, pd.DataFrame]) -> dict:
    """AUROC on LGG of mRNA, methylation as loaded, methylation aligned with mRNA, and permuted labels."""
    X_mrna, y, _ = mlomics.load_task(f"{CANCER}/mrna", root)
    X_omic, y_omic, _ = mlomics.load_task(f"{CANCER}/{OMIC}", root)
    report = {
        "mrna": cross_validated_auroc(X_mrna, y),
        f"{OMIC}_as_loaded": cross_validated_auroc(X_omic, y_omic),
        f"{OMIC}_permuted_labels": cross_validated_auroc(X_omic, np.random.default_rng(SEED).permutation(y_omic)),
    }
    # Methylation rows as loaded follow the columns of its file; put them in the order of mRNA, whose position the
    # labels follow, keeping the patients present in both.
    rows = {p: i for i, p in enumerate(patient(s) for s in tables[OMIC].columns)}
    order = [(i, rows[patient(s)]) for i, s in enumerate(tables["mrna"].columns) if patient(s) in rows]
    if len(rows) == len(tables[OMIC].columns) and order:
        label_rows, omic_rows = map(np.array, zip(*order, strict=True))
        report[f"{OMIC}_in_mrna_order"] = cross_validated_auroc(X_omic[omic_rows], y[label_rows])
        report[f"{OMIC}_in_mrna_order_samples"] = len(order)
    else:
        report[f"{OMIC}_in_mrna_order"] = None  # duplicated patients: no unambiguous alignment
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    args = parser.parse_args()
    root = data_root(args)
    report = {"samples": {}, "combined_task_follows_mrna_order": {}, "labels": {}, "methylation_features": {}}
    lgg_tables = None
    for cancer in mlomics.CANCERS:
        tables = {omic: raw_omic(root, cancer, omic) for omic in mlomics.OMIC_FILES}
        report["samples"][cancer] = samples_report(tables)
        report["combined_task_follows_mrna_order"][cancer] = combined_follows_mrna(tables)
        report["labels"][cancer] = labels_report(root, cancer, tables["mrna"].shape[1])
        report["methylation_features"][cancer] = features_report(tables[OMIC])
        if cancer == CANCER:
            lgg_tables = tables
        print(f"{cancer}: checked", flush=True)
    report["predictability_auroc"] = predictability(root, lgg_tables)
    output = RESULTS / "audit" / "lgg_methylation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
