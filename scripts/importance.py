# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Feature importance of each method of the paper, scored the same way (scripts 08 and 09).

Every method is fitted on the same training rows and returns one score per column (higher means more important; NaN
for a column it dropped) and, if it predicts, the class probabilities of the query rows:

- tabomics-v2.5:n1: TabOmics v2.5, one member, the attention of the label tokens of the query rows (the default);
- tabomics-v2.5:n1:allrows: the same, averaged over all the rows, training and query, as TabPFN-Wide;
- tabpfn-wide-5k:n1: TabPFN-Wide, its 5k checkpoint, one member and one feature per group, the attention of the label
  token averaged over all the rows (its default);
- random_forest: the impurity importance of the random forest of the benchmarks;
- anova: the ANOVA F statistic of each column (no predictions).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from tabomics_paper.baselines import SEED
from tabomics_paper.weights import weights_path

METHODS = ("tabomics-v2.5:n1", "tabomics-v2.5:n1:allrows", "tabpfn-wide-5k:n1", "random_forest", "anova")

# The 50 genes of the PAM50 classifier of breast cancer subtypes (Parker et al., 2009), which assign the breast labels
# of MLOmics and Shamir; and older names of some of them that expression tables still use.
PAM50 = frozenset(("ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6", "CDH3",
                   "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160",
                   "GRB7", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67", "MLPH", "MMP11",
                   "MYBL2", "MYC", "NAT1", "NDC80", "NUF2", "ORC6", "PGR", "PHGDH", "PTTG1", "RRM2", "SFRP1",
                   "SLC39A6", "TMEM45B", "TYMS", "UBE2C", "UBE2T"))  # fmt: skip
PAM50_ALIASES = {"ORC6L": "ORC6", "KNTC2": "NDC80", "CDCA1": "NUF2"}


def fit_and_score(method: str, X_train, y_train, X_query, device: str = "cuda"):
    """Importance scores of a method fitted on (X_train, y_train) with X_query as query rows, and the class
    probabilities of X_query (None for anova)."""
    X_train, X_query = np.asarray(X_train, dtype=np.float32), np.asarray(X_query, dtype=np.float32)
    if method == "anova":
        from sklearn.feature_selection import f_classif

        with np.errstate(divide="ignore", invalid="ignore"):
            f = f_classif(X_train, y_train)[0]
        return np.where(np.isfinite(f), f, np.nan), None
    if method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        forest = RandomForestClassifier(n_jobs=-1, random_state=SEED).fit(X_train, y_train)
        return forest.feature_importances_, forest.predict_proba(X_query)
    if method == "tabpfn-wide-5k:n1":
        from tabpfnwide.classifier import TabPFNWideClassifier

        wide = TabPFNWideClassifier(
            model_path=str(weights_path("tabpfn-wide-5k")),
            features_per_group=1,
            n_estimators=1,
            random_state=SEED,
            device=device,
            save_attention_maps="label",
        )
        proba = wide.fit(X_train, y_train).predict_proba(X_query)  # the forward pass whose attention is read
        return np.asarray(wide.get_attention_to_label(), dtype=float), np.asarray(proba)
    from tabomics import Classifier

    rows = {"tabomics-v2.5:n1": "test", "tabomics-v2.5:n1:allrows": "all"}[method]
    model = Classifier(
        model="v2.5", device=device, categorical_features=[], infer_categorical=False, n_estimators=1, random_state=SEED
    ).fit(X_train, y_train)
    return model.feature_importance(X_query, rows=rows), model.predict_proba(X_query)


def ranked(scores) -> np.ndarray:
    """Scores with NaN (a column a method dropped) below every scored column."""
    scores = np.asarray(scores, dtype=float)
    lowest = np.nanmin(scores) - 1.0 if np.isfinite(scores).any() else 0.0
    return np.where(np.isnan(scores), lowest, scores)


def top(scores, k: int) -> np.ndarray:
    """Indices of the k highest scores."""
    return np.argsort(-ranked(scores), kind="stable")[:k]


def recovery_auroc(scores, informative) -> float:
    """Probability that an informative column scores above a noise column, ties counted as one half."""
    is_informative = np.zeros(len(scores), dtype=bool)
    is_informative[np.asarray(informative)] = True
    return float(roc_auc_score(is_informative, ranked(scores)))


def recall_at_k(scores, informative) -> float:
    """Share of the k informative columns among the k top-scored ones (the Recall@k of TabPFN-Wide)."""
    informative = set(np.asarray(informative).tolist())
    return len(set(top(scores, len(informative)).tolist()) & informative) / len(informative)


def gene_symbol(column: str) -> str:
    """The gene symbol of a column name: 'ESR1', 'ESR1|2099' -> 'ESR1', older PAM50 names as the current ones."""
    symbol = str(column).split("|")[0].strip().upper()
    return PAM50_ALIASES.get(symbol, symbol)
