from __future__ import annotations

import importlib
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
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _spec():
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=5,
        n_items=3,
        seed=2401,
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


def test_prior_object_classes_sampling_and_declared_draws(monkeypatch):
    spec = _spec()
    table = pb.prior_specification_table(spec)
    assert not table.empty
    assert pb._prior_object(spec) is spec.priors
    assert pb._prior_object(spec.priors) is spec.priors

    with pytest.raises(GP3BayesError):
        pb._prior_object(object())

    assert pb._class_for_variable("Intercept") == "Intercept"
    assert pb._class_for_variable("b_Intercept") == "Intercept"
    assert pb._class_for_variable("b_x") == "b"
    assert pb._class_for_variable("sd_participant") == "sd"
    assert pb._class_for_variable("cor_participant__x") == "cor"
    assert pb._class_for_variable("sigma") == "sigma"
    assert pb._class_for_variable("latent") is None

    rng = np.random.default_rng(1)
    normal = pd.Series(
        {
            "distribution": "normal",
            "location": 0.0,
            "scale": 1.0,
            "df": np.nan,
            "lower": np.nan,
            "shape": np.nan,
        }
    )
    student = pd.Series(
        {
            "distribution": "student_t",
            "location": 0.0,
            "scale": 1.0,
            "df": 4.0,
            "lower": 0.0,
            "shape": np.nan,
        }
    )
    lkj = pd.Series(
        {
            "distribution": "lkj",
            "location": np.nan,
            "scale": np.nan,
            "df": np.nan,
            "lower": np.nan,
            "shape": 2.0,
        }
    )
    assert len(pb._sample_prior(normal, 50, rng)) == 50
    assert (pb._sample_prior(student, 50, rng) >= 0).all()
    assert np.all(np.abs(pb._sample_prior(lkj, 50, rng)) <= 1)

    bad = normal.copy()
    bad["distribution"] = "mystery"
    with pytest.raises(GP3BayesError):
        pb._sample_prior(bad, 50, rng)

    class_to_variable = {
        "Intercept": "b_Intercept",
        "b": "b_x",
        "sd": "sd_participant",
        "cor": "cor_participant__x",
    }
    variables = tuple(
        class_to_variable[cls] for cls in table["parameter_class"] if cls in class_to_variable
    )
    variables = tuple(dict.fromkeys(variables))
    draws = pb.simulate_declared_prior_draws(
        spec,
        variables=variables,
        ndraws=50,
        seed=2,
    )
    assert set(draws.columns) == set(variables)

    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(spec, variables=variables, ndraws=49)
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(spec, variables=None)
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            spec,
            variables=variables,
            regex=r"^never$",
            ndraws=50,
        )
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            spec,
            variables=("latent",),
            ndraws=50,
        )

    fit = SimpleNamespace(
        fit_performed=True,
        specification=spec,
        family="binary",
    )
    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "b_Intercept": np.linspace(-1, 1, 60),
                "b_x": np.linspace(0.5, -0.5, 60),
            }
        ),
    )
    inferred = pb.simulate_declared_prior_draws(
        fit,
        variables=None,
        ndraws=50,
        seed=3,
    )
    assert set(inferred.columns) == {"b_Intercept", "b_x"}


def test_prior_posterior_bridge_tables_long_and_plots(monkeypatch):
    spec = _spec()
    fit = SimpleNamespace(
        fit_performed=True,
        specification=spec,
        family="binary",
    )

    posterior = pd.DataFrame(
        {
            ".chain": np.repeat([1, 2], 40),
            ".iteration": np.tile(np.arange(1, 41), 2),
            ".draw": np.arange(1, 81),
            "b_Intercept": np.linspace(-0.4, 0.5, 80),
            "b_x": np.linspace(-0.2, 0.3, 80),
            "latent": np.linspace(0, 1, 80),
        }
    )
    monkeypatch.setattr(
        pb,
        "extract_posterior_draws",
        lambda *args, **kwargs: posterior.copy(),
    )

    bridge = pb.prior_posterior_bridge(
        fit,
        ndraws=60,
        probs=(0.1, 0.5, 0.9),
        seed=4,
    )
    assert bridge.family == "binary"
    assert bridge.variables == ("b_Intercept", "b_x")
    assert len(pb.prior_posterior_summary_table(bridge)) == 2
    assert len(pb.prior_posterior_distance_table(bridge)) == 2
    assert bridge.to_frame().equals(bridge.summary)

    long = pb.prior_posterior_draws_long(
        bridge,
        max_draws=50,
        seed=5,
    )
    assert set(long["distribution"]) == {"prior", "posterior"}
    assert long["draw"].max() == 50

    assert pb.plot_prior_posterior_density(bridge, max_draws=50).axes
    assert pb.plot_prior_posterior_intervals(bridge).axes
    assert pb.plot_prior_posterior_shift(bridge).axes
    assert pb.plot_prior_posterior_contraction(bridge).axes

    with pytest.raises(GP3BayesError):
        pb.prior_posterior_summary_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_distance_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_draws_long(object(), max_draws=50)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_draws_long(bridge, max_draws=49)
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_bridge(
            fit,
            probs=(0.5, 0.1, 0.9),
        )

    assert np.isnan(pb._overlap(0.0, 0.0, 0.0, 0.0))
