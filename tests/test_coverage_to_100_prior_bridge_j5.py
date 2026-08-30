from __future__ import annotations

import importlib
import re
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

pb = importlib.import_module("gp3bayespy.prior_posterior_bridge")

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _spec(seed=4901):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return gp.specify_binary_model(prepared)


def test_prior_bridge_remaining_sampling_and_full_bridge(monkeypatch):
    spec = _spec()
    priors = spec.priors

    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            priors,
            variables=("b_Intercept",),
            ndraws=49,
        )
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            priors,
            variables=("b_Intercept",),
            seed=-1,
        )
    with pytest.raises(re.error):
        pb.simulate_declared_prior_draws(
            priors,
            variables=("b_Intercept",),
            regex="[",
            ndraws=50,
        )

    student = pd.Series(
        {
            "distribution": "student_t",
            "location": 0,
            "scale": 1,
            "df": 3,
            "lower": -np.inf,
            "shape": np.nan,
        }
    )
    rng = np.random.default_rng(3)
    draws = pb._sample_prior(student, 100, rng)
    assert np.any(draws < 0)

    posterior = pd.DataFrame(
        {
            "b_Intercept": np.linspace(-0.5, 0.5, 120),
            "unsupported": np.ones(120),
        }
    )
    fit = SimpleNamespace(
        fit_performed=True,
        family="binary",
        specification=SimpleNamespace(priors=priors),
    )

    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *args, **kwargs: posterior.copy(),
    )
    bridge = pb.prior_posterior_bridge(fit, ndraws=50, seed=4)
    assert bridge.variables == ("b_Intercept",)
    assert len(bridge.posterior_draws) == 50
    assert not bridge.summary.empty
    assert not bridge.distances.empty

    with pytest.raises(GP3BayesError):
        pb.prior_posterior_bridge(fit, probs=(0.5, 0.4, 0.9))

    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *args, **kwargs: pd.DataFrame({"unsupported": [1.0, 2.0]}),
    )
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_bridge(fit, ndraws=50)

    prior = pd.DataFrame({"b_Intercept": np.linspace(-1, 1, 120)})
    post = pd.DataFrame({"b_Intercept": np.linspace(-0.5, 0.5, 120)})
    summary = bridge.summary.copy()
    distances = bridge.distances.copy()
    long_bridge = pb.PriorPosteriorBridge(
        "0.3",
        "binary",
        ("b_Intercept",),
        prior,
        post,
        summary,
        distances,
        (0.025, 0.5, 0.975),
        1,
    )
    long = pb.prior_posterior_draws_long(long_bridge, max_draws=50, seed=5)
    assert len(long) == 100
