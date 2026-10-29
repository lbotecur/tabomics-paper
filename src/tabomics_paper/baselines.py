# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""The baseline models, by name. Each is built for one fold and only needs fit(X, y) and predict_proba(X).

Every model gets the same data: a float matrix whose categorical columns hold integer codes (NaN for missing values)
and the list of those columns; each model is told about them in its own way. Their packages, at the versions of the
paper, are dependencies of this repository (pyproject.toml).

Brought from tabomics 0.1.0 (baselines/models.py), where the baselines were kept until 0.2.0.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabomics_paper.weights import weights_path

# Classical models, without hyperparameter search, as TabPFN-Wide evaluates them.
CLASSIC = ("random_forest", "logreg_en", "xgboost")
# The other tabular foundation models of the paper, each with its fixed weights (tabomics_paper.weights). A suffix
# ":n<k>" sets the ensemble members (no suffix: 1, a single forward pass); everything else keeps the package's
# defaults. TabPFN-3.5 sees at most 768 features per member, so one member only sees a subset of a wide table.
FOUNDATION = ("tabpfn-3.5", "tabpfn-v2-gn2p4bpt", "tabpfn-wide-5k", "tabicl-v2")
SEED = 42


def parse_name(name: str) -> tuple[str, int]:
    """'tabpfn-3.5:n8' -> ('tabpfn-3.5', 8); no suffix -> 1."""
    base, _, suffix = name.partition(":")
    return base, int(suffix.removeprefix("n")) if suffix else 1


def available() -> list[str]:
    """Names accepted by make_model (the foundation models also with ":n<k>")."""
    return [*CLASSIC, *FOUNDATION]


def elastic_net_cv(categorical: list[int]) -> GridSearchCV:
    """Elastic-net logistic regression (l1_ratio 0.5), C among five values from 1e-4 to 1e4 chosen by 3-fold CV on
    the training rows with the log-loss, then refitted on all of them. The imputation and standardization are part of
    the pipeline, so that they are fitted within each partition of the cross-validation. saga stops when it converges;
    max_iter 5000, as the other logistic regressions of the repository, is well above what every fit needs (at most
    about 600 iterations on the tasks checked), so that no fit stops before converging."""
    penalty = {} if tuple(map(int, sklearn.__version__.split(".")[:2])) >= (1, 8) else {"penalty": "elasticnet"}
    model = LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000, tol=1e-3, random_state=SEED, **penalty)
    return GridSearchCV(
        make_pipeline(linear_inputs(categorical), model),
        {"logisticregression__C": np.logspace(-4, 4, 5)},
        cv=StratifiedKFold(3),
        scoring="neg_log_loss",
        n_jobs=-1,
    )


def linear_inputs(categorical: list[int]):
    """For linear models: one-hot categorical columns; numeric columns imputed (median) and standardized."""
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    if not categorical:
        return numeric
    onehot = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    return ColumnTransformer([("cat", onehot, list(categorical))], remainder=numeric)


def make_model(name: str, categorical: list[int], device: str = "cuda"):
    """A fresh estimator for one fold.

    Args:
        name: One of available(), the foundation models optionally with ":n<k>".
        categorical: Indices of the categorical columns of the task ([] in the benchmarks of the paper).
        device: Where the models that can use a GPU run.
    """
    if name == "random_forest":  # as TabPFN-Wide: default settings, missing values imputed with the mode
        return make_pipeline(
            SimpleImputer(strategy="most_frequent"), RandomForestClassifier(n_jobs=-1, random_state=SEED)
        )
    if name == "logreg_en":
        return elastic_net_cv(categorical)
    if name == "xgboost":  # as TabPFN-Wide: 1000 trees, histogram method, on the GPU if given
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=1000, tree_method="hist", device=device, random_state=SEED, n_jobs=-1)

    base, n_estimators = parse_name(name)
    if base not in FOUNDATION:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(available())}")
    model_path = str(weights_path(base))
    if base in ("tabpfn-3.5", "tabpfn-v2-gn2p4bpt"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version = ModelVersion.V3_5 if base == "tabpfn-3.5" else ModelVersion.V2
        return TabPFNClassifier.create_default_for_version(
            version,  # the defaults of that version of TabPFN ...
            model_path=model_path,  # ... with these weights
            n_estimators=n_estimators,
            random_state=SEED,
            device=device,
            ignore_pretraining_limits=True,  # wide tables are beyond its pretraining limits
            categorical_features_indices=list(categorical),
        )
    if base == "tabpfn-wide-5k":
        from tabpfnwide.classifier import TabPFNWideClassifier

        return TabPFNWideClassifier(
            model_path=model_path,
            features_per_group=1,  # its default: one feature per token, as TabPFN-Wide is trained and evaluated
            n_estimators=n_estimators,
            random_state=SEED,
            device=device,
            categorical_features_indices=list(categorical),
        )
    from tabicl import TabICLClassifier  # tabicl-v2; TabICL takes no list of categorical columns

    return TabICLClassifier(
        model_path=model_path,
        allow_auto_download=False,
        n_estimators=n_estimators,
        random_state=SEED,
        device=device,
    )
