from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import BackendUnavailableError, GP3BayesError

po = importlib.import_module("gp3bayespy.posterior")
pb = importlib.import_module("gp3bayespy.prior_posterior_bridge")


class PosteriorContainer:
    def __init__(self):
        self.data_vars = {"x": np.ones((2, 3))}


def _binary_spec(seed=5501):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=3,
        trials_per_participant=5,
        n_items=2,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("trial_covariate",),
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return gp.specify_binary_model(prepared)


def test_posterior_data_vars_attribute_branch():
    fit = SimpleNamespace(backend_fit=SimpleNamespace(posterior=PosteriorContainer()))
    assert "x" in po._posterior_data_vars(fit)


def test_posterior_optional_backend_error_branches(monkeypatch):
    selected = {"b_x": np.ones((2, 3))}
    monkeypatch.setattr(po, "_select_components", lambda *a, **k: selected)
    real_import = po.import_module

    def no_xarray(name):
        if name == "xarray":
            raise ImportError("synthetic xarray absence")
        return real_import(name)

    monkeypatch.setattr(po, "import_module", no_xarray)
    with pytest.raises(BackendUnavailableError):
        po.extract_draws(object(), format="array")

    def no_arviz(name):
        if name == "arviz":
            raise ImportError("synthetic arviz absence")
        return real_import(name)

    monkeypatch.setattr(po, "import_module", no_arviz)
    with pytest.raises(BackendUnavailableError):
        po._arviz()

    monkeypatch.setattr(po, "_arviz", lambda: object())
    monkeypatch.setattr(po, "import_module", no_xarray)
    with pytest.raises(BackendUnavailableError):
        po._diagnostic_metrics(selected)


def test_posterior_dataset_scalar_exception_and_sample_stats_second_name():
    assert np.isnan(po._dataset_scalar(object(), "x"))
    arr = po._sample_stats_array(
        {"second": np.ones((2, 3))},
        ("first", "second"),
    )
    assert arr is not None and arr.shape == (2, 3)


def test_prior_bridge_ndarray_inference_unique_prior_and_no_subsample(monkeypatch):
    spec = _binary_spec()
    fit = SimpleNamespace(
        fit_performed=True,
        family="binary",
        specification=spec,
    )

    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *a, **k: np.ones((10, 2)),
    )
    with pytest.raises(GP3BayesError, match="variable names"):
        pb.simulate_declared_prior_draws(
            fit,
            variables=None,
            ndraws=50,
            seed=1,
        )

    with pytest.raises(GP3BayesError, match="No unique declared prior"):
        pb.simulate_declared_prior_draws(
            spec.priors,
            variables=("sigma",),
            ndraws=50,
            seed=1,
        )

    prior = pd.DataFrame({"b_Intercept": np.linspace(-1, 1, 50)})
    posterior = pd.DataFrame({"b_Intercept": np.linspace(-0.5, 0.5, 50)})
    bridge = pb.PriorPosteriorBridge(
        "0.3",
        "binary",
        ("b_Intercept",),
        prior,
        posterior,
        pd.DataFrame(),
        pd.DataFrame(),
        (0.025, 0.5, 0.975),
        1,
    )
    long = pb.prior_posterior_draws_long(
        bridge,
        max_draws=50,
        seed=1,
    )
    assert len(long) == 100
