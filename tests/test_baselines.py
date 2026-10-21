"""The baselines: names, and the classical models (brought from tabomics 0.1.0, with its tests)."""

import numpy as np
import pytest

from tabomics_paper import baselines as models


def test_names():
    assert models.parse_name("tabpfn-3.5:n8") == ("tabpfn-3.5", 8)
    assert models.parse_name("tabpfn-3.5:cover") == ("tabpfn-3.5", "cover")
    assert models.parse_name("tabicl-v2") == ("tabicl-v2", 1)
    with pytest.raises(ValueError, match="Unknown model"):
        models.make_model("svm", [])


@pytest.mark.parametrize("name", ["random_forest", "logreg", "logreg_en"])
def test_classical_models_fit_with_missing_and_categorical_values(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    X[:, 1] = rng.integers(0, 3, 60)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int)
    proba = models.make_model(name, [1]).fit(X[:40], y[:40]).predict_proba(X[40:])
    assert proba.shape == (20, 2)
