# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""10. Sample embeddings on the mRNA tasks of MLOmics and Shamir (results/raw/embeddings/).

    python scripts/10_embeddings.py --data-root /path/to/data
    python scripts/10_embeddings.py --models tabomics-v2.5:n8 --tasks mlomics/BRCA/mrna   # one model and task

The protocol of Section 6.6 of the paper, on the 8 mRNA tasks and their outer folds (scripts/02_folds.py). Each
representation of the samples is built from the outer training rows only:

- tabomics-v2.5:n1 and tabomics-v2.5:n8 (the members averaged, with the class indices fixed), tabpfn-wide-5k:n1 and
  tabpfn-v2-gn2p4bpt:n1 (TabPFNv2): the representation of the label token after the last block, as in the benchmarks.
  The training rows are embedded out of fold (Ye et al., 2025): in ten stratified parts, each as test rows with the
  other nine as context, so that no row is embedded with its own label; the test rows with all the training rows as
  context. TabPFN-Wide and TabPFNv2 with one member: their members permute the class indices, and their package
  returns one embedding per member.
- all: the genes; pca: their principal components, as many as the size of the embeddings of TabOmics (192) if the
  rows allow it; pls: 2, 5, 10 or 20 partial least squares components of the genes and the classes. The genes are
  standardized first.

Every representation is standardized and given to the same two classifiers, a logistic regression (l2) and a
k-nearest-neighbor classifier (Euclidean, uniform weights), scored on the outer test rows (AUROC, accuracy). Their
hyperparameters (C, five values from 1e-4 to 1e4; k in 1, 3, 5, 11; and the PLS components) are chosen on the outer
training rows by stratified 3-fold cross-validation with the macro AUROC: on the embeddings of the outer training rows,
computed once per outer fold; for the genes, with the scaling, PCA and PLS fitted within each inner partition. No
label of the outer test rows enters a representation or the choice of a hyperparameter.

One CSV per model and task, results/raw/embeddings/<model>/<suite>_<cancer>.csv, one row per fold and classifier,
written after every fold: a run that stops resumes where it left off. With --figure, only figures/embeddings.pdf,
from the first fold of Shamir breast: for the genes and the embeddings of the four foundation models, the out-of-fold
representation of the training rows on its first two principal components, by PAM50 subtype, one panel each; the
representations are kept in FIGURE_CACHE, outside the repository, so that the figure is drawn again in seconds. Only
scores and a figure are written to the repository, never data. One GPU, chosen with CUDA_VISIBLE_DEVICES.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabomics_paper.baselines import SEED, make_model, parse_name
from tabomics_paper.benchmarks import mlomics, shamir
from tabomics_paper.benchmarks.metrics import accuracy, roc_auc

from common import REPO, RESULTS, add_data_root, data_root, paper_name

SUITES = {"mlomics": mlomics, "shamir": shamir}
TASKS = [f"mlomics/{cancer}/mrna" for cancer in mlomics.CANCERS] + [f"shamir/{c}/mrna" for c in shamir.LABEL_COLUMNS]
MODELS = ("tabomics-v2.5:n1", "tabomics-v2.5:n8", "tabpfn-wide-5k:n1", "tabpfn-v2-gn2p4bpt:n1", "all", "pca", "pls")
GENES = ("all", "pca", "pls")  # the representations without a foundation model
EMBEDDING_SIZE = 192  # of TabOmics v2.5, the components of pca
PLS_COMPONENTS = (2, 5, 10, 20)
CLASSIFIERS = {"logreg": [{"C": c} for c in np.logspace(-4, 4, 5)], "knn": [{"n_neighbors": k} for k in (1, 3, 5, 11)]}
PARTS, INNER_FOLDS = 10, 3
FIGURE = ("shamir/breast/mrna", 0)
# Its classes in the order of their codes: the PAM50 subtypes of Shamir are text, coded in alphabetical order (those of
# MLOmics are numbers whose subtypes are not documented, hence not this task).
# Colors of the Okabe-Ito palette, distinguishable with color-vision deficiencies.
FIGURE_CLASSES = {"Basal": "#0072B2", "Her2": "#E69F00", "LumA": "#009E73", "LumB": "#D55E00", "Normal": "#CC79A7"}
FIGURE_MODELS = ("all", "tabomics-v2.5:n1", "tabomics-v2.5:n8", "tabpfn-wide-5k:n1", "tabpfn-v2-gn2p4bpt:n1")
FIGURE_PATH = REPO / "figures" / "embeddings.pdf"
# The representations drawn, kept outside the repository (they are per sample, never committed), so that the figure
# can be drawn again in seconds; delete the file to compute them again.
FIGURE_CACHE = Path.home() / ".cache" / "tabomics-paper" / "embeddings_figure.npz"
OUTPUT = RESULTS / "raw" / "embeddings"


def embedder(model: str, device: str):
    """A function that fits the model on (X, y) and returns the embeddings of X_query."""
    base, passes = parse_name(model)
    if base == "tabomics-v2.5":
        from tabomics import Classifier

        return lambda X, y, X_query: Classifier(model="v2.5", device=device, categorical_features=[],
                                                infer_categorical=False, n_estimators=passes,
                                                random_state=SEED).fit(X, y).embeddings(X_query)  # fmt: skip
    return lambda X, y, X_query: make_model(model, [], device).fit(X, y).get_embeddings(X_query)[0]


def represent(model, embed, X, y, train, query, components=None):
    """The representation of the training and query rows, built from the training rows only."""
    if embed is not None:
        parts = list(StratifiedKFold(PARTS).split(train, y[train]))
        Z = None
        for context, part in parts:
            assert set(y[train[context]]) == set(y[train]), "a context without a class encodes the labels otherwise"
            z = embed(X[train[context]], y[train[context]], X[train[part]])
            Z = np.empty((len(train), z.shape[1])) if Z is None else Z
            Z[part] = z
        return Z, embed(X[train], y[train], X[query])
    scaler = StandardScaler().fit(X[train])
    A, B = scaler.transform(X[train]), scaler.transform(X[query])
    if model == "pca":
        pca = PCA(min(EMBEDDING_SIZE, len(train) - 1, X.shape[1]), random_state=SEED).fit(A)
        return pca.transform(A), pca.transform(B)
    if model == "pls":
        pls = PLSRegression(components, scale=False).fit(A, np.eye(y.max() + 1)[y[train]])
        return pls.transform(A), pls.transform(B)
    return A, B


def classifier(name: str, params: dict):
    estimator = LogisticRegression(max_iter=5000, random_state=SEED, **params) if name == "logreg" else \
        KNeighborsClassifier(**params)  # fmt: skip
    return make_pipeline(StandardScaler(), estimator)


def run_fold(model, embed, X, y, train, test):
    """The rows of one outer fold, and the representation of its training rows (out of fold for the embeddings)."""
    inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED).split(train, y[train]))
    assert all(set(y[train[a]]) == set(y[train]) for a, _ in inner), "an inner training partition without a class"
    outer = {} if embed is None else {None: represent(model, embed, X, y, train, test)}  # the embeddings, once
    scores = []  # (inner AUROC, classifier, components, hyperparameters)
    for components in PLS_COMPONENTS if model == "pls" else (None,):
        if embed is None:  # scaling, PCA and PLS fitted within each inner training partition
            partitions = [represent(model, None, X, y, train[a], train[b], components) for a, b in inner]
        else:  # the out-of-fold embeddings of the outer training rows
            partitions = [(outer[None][0][a], outer[None][0][b]) for a, b in inner]
        partitions = [(A, B, y[train[a]], y[train[b]]) for (A, B), (a, b) in zip(partitions, inner, strict=True)]
        for name, grid in CLASSIFIERS.items():
            for params in grid:
                auroc = [roc_auc(yb, classifier(name, params).fit(A, ya).predict_proba(B))
                         for A, B, ya, yb in partitions]  # fmt: skip
                scores.append((np.nanmean(auroc), name, components, params))
    rows = []
    for name in CLASSIFIERS:
        _, _, components, params = max((s for s in scores if s[1] == name), key=lambda s: s[0])
        if components not in outer:
            outer[components] = represent(model, embed, X, y, train, test, components)
        A, B = outer[components]
        proba = classifier(name, params).fit(A, y[train]).predict_proba(B)
        rows.append({"classifier": name, "roc_auc": roc_auc(y[test], proba), "accuracy": accuracy(y[test], proba),
                     "hyperparameter": next(iter(params.values())), "components": components,
                     "dimension": A.shape[1]})  # fmt: skip
    return rows, next(iter(outer.values()))[0]


def figure(panels: list[tuple[str, np.ndarray]], y: np.ndarray) -> None:
    """One panel per representation: the training rows on its first two principal components, by subtype; one legend."""
    fig, axes = plt.subplots(1, len(panels), figsize=(1.45 * len(panels), 1.5))
    for ax, (model, Z) in zip(axes, panels, strict=True):
        points = PCA(2, random_state=SEED).fit_transform(StandardScaler().fit_transform(Z))
        for code, (name, color) in enumerate(FIGURE_CLASSES.items()):
            ax.scatter(*points[y == code].T, s=2.5, color=color, linewidths=0, label=name)
        ax.set(xticks=[], yticks=[], box_aspect=1)
        ax.set_title(paper_name(model), fontsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.08),
               ncol=len(FIGURE_CLASSES), fontsize=7, frameon=False, markerscale=3, handletextpad=0.1)  # fmt: skip
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight", metadata={"CreationDate": None})  # the same bytes on every run
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_data_root(parser)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--figure", action="store_true", help="only the figure (FIGURE_PATH), from its fold alone")
    args = parser.parse_args()
    root = data_root(args)
    if args.figure:  # the out-of-fold representations of the training rows of that fold; no CSV is written
        if not FIGURE_CACHE.exists():
            suite, task = FIGURE[0].split("/", 1)
            X, y, _ = SUITES[suite].load_task(task, root)
            train, test = SUITES[suite].folds(y)[FIGURE[1]]
            representations = {"y": y[train]}
            for i, model in enumerate(FIGURE_MODELS):
                embed = None if model in GENES else embedder(model, args.device)
                representations[f"panel{i}"] = represent(model, embed, X, y, train, test)[0]
                print(f"{model}: represented", flush=True)
            FIGURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(FIGURE_CACHE, **representations)
        cached = np.load(FIGURE_CACHE)
        figure([(model, cached[f"panel{i}"]) for i, model in enumerate(FIGURE_MODELS)], cached["y"])
        print(f"written: {FIGURE_PATH.relative_to(REPO)}", flush=True)
        return 0
    for model in args.models:
        embed = None if model in GENES else embedder(model, args.device)
        for name in args.tasks:
            suite, task = name.split("/", 1)
            module = SUITES[suite]
            path = OUTPUT / model.replace(":", "_") / f"{suite}_{task.split('/')[0]}.csv"
            results = pd.read_csv(path) if path.exists() else pd.DataFrame()
            X, y, _ = module.load_task(task, root)
            for fold, (train, test) in enumerate(module.folds(y)):
                if len(results) and fold in set(results.fold):
                    continue
                rows, _ = run_fold(model, embed, X, y, train, test)
                fold_rows = pd.DataFrame(rows).assign(model=model, suite=suite, task=task, fold=fold)
                results = pd.concat([results, fold_rows], ignore_index=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(path, index=False)
                print(f"{model} | {name} fold {fold} | " + " | ".join(
                    f"{r['classifier']} AUROC {r['roc_auc']:.3f}" for r in rows), flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
