# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Test whether TabOmics embeddings conditioned on luminal status distinguish LumA from LumB."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common import RESULTS, add_data_root, data_root
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabomics_paper.baselines import SEED
from tabomics_paper.benchmarks import shamir

TASK = "breast/mrna"
LUMINAL = ("LumA", "LumB")
EXPECTED_LABELS = {"Basal", "Her2", "LumA", "LumB", "Normal"}
K = 10
PLS_COMPONENTS = 5
PRIMARY = "tabomics_coarse"
OUTPUT = RESULTS / "followups" / "coarse_luminal_embeddings.csv"


def representations(classifier, X_support: np.ndarray, y_support: np.ndarray, X_query: np.ndarray, seed: int):
    """Build all five representations from expression and the binary support labels."""
    classifier.fit(X_support, y_support)
    embeddings = classifier.embeddings(X_query)
    probability = classifier.predict_proba(X_query)[:, 1]

    permuted = np.random.default_rng(seed).permutation(y_support)
    classifier.fit(X_support, permuted)
    permuted_embeddings = classifier.embeddings(X_query)

    scaler = StandardScaler().fit(X_support)
    scaled_support = scaler.transform(X_support)
    expression = scaler.transform(X_query)
    pls = PLSRegression(n_components=PLS_COMPONENTS, scale=False).fit(scaled_support, y_support)
    return {
        PRIMARY: embeddings,
        "expression": expression,
        "pls_coarse": pls.transform(expression),
        "coarse_probability": probability[:, None],
        "tabomics_permuted": permuted_embeddings,
    }


def neighbor_scores(values: np.ndarray, subtypes: np.ndarray) -> dict[str, float]:
    """Score ten-neighbor purity within luminal patients and average equally across both subtypes."""
    dist = euclidean_distances(np.asarray(values, dtype=np.float64))
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argsort(dist, axis=1, kind="stable")[:, :K]
    purity = (subtypes[neighbors] == subtypes[:, None]).mean(axis=1)
    scores = {f"purity_{name}": float(purity[subtypes == name].mean()) for name in LUMINAL}
    return {**scores, "purity_macro": float(np.mean(list(scores.values())))}


def evaluate(X: np.ndarray, subtypes: np.ndarray, classifier, seeds: list[int]) -> pd.DataFrame:
    """Evaluate each support draw on the luminal patients of its held-out paper fold."""
    if set(subtypes) != EXPECTED_LABELS:
        raise ValueError(f"Expected the five PAM50 labels {sorted(EXPECTED_LABELS)}, got {sorted(set(subtypes))}")
    coarse = np.isin(subtypes, LUMINAL).astype(int)
    rows = []
    for fold, (outer_train, query) in enumerate(shamir.folds(LabelEncoder().fit_transform(subtypes))):
        keep = coarse[query] == 1
        query_subtypes = subtypes[query][keep]
        counts = {f"n_{name}": int((query_subtypes == name).sum()) for name in LUMINAL}
        if len(query_subtypes) <= K or min(counts.values()) == 0:
            raise ValueError(f"Fold {fold} needs both subtypes and at least {K + 1} luminal patients, got {counts}")
        for seed in seeds:
            support, _ = train_test_split(outer_train, train_size=0.5, random_state=seed, stratify=coarse[outer_train])
            support.sort()
            A, B, _ = shamir.prepare(X, support, query)
            spaces = representations(classifier, A, coarse[support], B, seed)
            info = {
                "fold": fold,
                "seed": seed,
                "n_support": len(support),
                "n_support_luminal": int(coarse[support].sum()),
                "n_query": len(query),
                **counts,
                "coarse_auroc": roc_auc_score(coarse[query], spaces["coarse_probability"].ravel()),
            }
            for method, values in spaces.items():
                rows.append({**info, "method": method, **neighbor_scores(values[keep], query_subtypes)})
            print(f"Completed fold {fold}, support seed {seed}", flush=True)
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> None:
    """Report means, paired differences, and variation across folds and support draws."""
    paired = results.pivot(index=["fold", "seed"], columns="method", values="purity_macro")
    deltas = paired.drop(columns=PRIMARY).rsub(paired[PRIMARY], axis=0)
    means = results.groupby("method")[["purity_LumA", "purity_LumB", "purity_macro"]].mean()
    means["mean_tabomics_minus_control"] = deltas.mean()
    means["runs_tabomics_wins"] = (deltas > 0).sum()
    print("\nMean neighborhood purity")
    print(means.round(4).to_string(na_rep="—"))
    print("\nTabOmics minus each control by fold, averaged over support draws")
    print(deltas.groupby("fold").mean().round(4).to_string())
    print("\nStandard deviation of paired differences across support draws, by fold")
    print(deltas.groupby("fold").std().round(4).to_string(na_rep="—"))
    print(f"Mean coarse-label AUROC: {results.loc[results.method == PRIMARY, 'coarse_auroc'].mean():.4f}")


def main() -> int:
    """Run the follow-up and write its aggregate CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="support draw seeds")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="aggregate results CSV")
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain distinct nonnegative integers")
    root = data_root(args)

    from tabomics import Classifier

    X, subtypes = shamir.load_labeled_task(TASK, root)
    classifier = Classifier(model="v2.5", device=args.device, categorical_features=[],
                            infer_categorical=False, n_estimators=1, random_state=SEED)  # fmt: skip
    results = evaluate(X, subtypes, classifier, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    summary(results)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
