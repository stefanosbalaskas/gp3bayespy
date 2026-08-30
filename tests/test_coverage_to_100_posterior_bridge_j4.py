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

matplotlib.use("Agg")

po = importlib.import_module("gp3bayespy.posterior")
pb = importlib.import_module("gp3bayespy.prior_posterior_bridge")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _arr(values):
    return SimpleNamespace(values=np.asarray(values, dtype=float))


def _fake_fit():
    posterior = {
        "b_Intercept": _arr(np.arange(6).reshape(2, 3)),
        "b": _arr(np.ones((2, 3, 1))),
        "sd_participant": _arr(np.ones((2, 3))),
        "sd_item": _arr(np.ones((2, 3)) * 0.5),
        "participant_chol_stds": _arr(np.ones((2, 3, 2))),
        "participant_chol_corr": _arr(np.tile(np.eye(2), (2, 3, 1, 1))),
        "sigma": _arr(np.ones((2, 3)) * 0.2),
        "extra": _arr(np.ones((2, 3, 2, 2))),
    }
    contract = SimpleNamespace(
        mappings={
            "participant": "pid",
            "item": "iid",
            "condition": "condition",
        }
    )
    prepared = SimpleNamespace(model_matrix_columns=("Intercept", "x"))
    specification = SimpleNamespace(
        prepared=prepared,
        contract=contract,
    )
    return SimpleNamespace(
        fit_performed=True,
        family="binary",
        backend_fit=SimpleNamespace(posterior=posterior),
        specification=specification,
    )


def _prior_spec():
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
        seed=4401,
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


def test_posterior_component_selection_and_format_matrix(monkeypatch):
    fit = _fake_fit()
    components = po._posterior_components(fit)
    assert "b_x" in components
    assert "sd_pid__Intercept" in components
    assert "sd_iid__Intercept" in components
    assert "cor_pid__Intercept__condition" in components
    assert any(name.startswith("extra[") for name in components)

    assert po._prepared_model_columns(SimpleNamespace()) == ()
    assert po._mapping_name(SimpleNamespace(), "item") is None
    assert po._component_name("x", (0, 2)) == "x[1,3]"

    selected = po._select_components(
        fit,
        parameters_only=True,
    )
    assert selected
    with pytest.raises(GP3BayesError):
        po._select_components(fit, variables=())
    with pytest.raises(GP3BayesError):
        po._select_components(fit, variables=("missing",))
    with pytest.raises(GP3BayesError):
        po._select_components(fit, regex="[")
    with pytest.raises(GP3BayesError):
        po._select_components(fit, regex="does-not-match")

    assert po.extract_draws(fit, variables=("b_Intercept",), format="matrix").shape == (6, 1)
    assert isinstance(
        po.extract_draws(fit, variables=("b_Intercept",), format="df"),
        pd.DataFrame,
    )
    assert isinstance(
        po.extract_draws(fit, variables=("b_Intercept",), format="rvars"),
        dict,
    )
    with pytest.raises(GP3BayesError):
        po.extract_draws(fit, format="bad")

    monkeypatch.setattr(
        po,
        "_select_components",
        lambda *args, **kwargs: {
            "a": np.zeros((2, 3)),
            "b": np.zeros((1, 3)),
        },
    )
    with pytest.raises(GP3BayesError):
        po.extract_draws(fit, format="matrix")


def test_posterior_validation_status_helpers_and_invalid_storage():
    with pytest.raises(GP3BayesError):
        po._validate_fit_like(SimpleNamespace())
    with pytest.raises(GP3BayesError):
        po._validate_fit_like(
            SimpleNamespace(
                fit_performed=True,
                family="pupil",
                backend_fit=SimpleNamespace(posterior={}),
            )
        )

    fit = _fake_fit()
    with pytest.raises(GP3BayesError):
        po._validate_fit_like(fit, "duration")

    bad = _fake_fit()
    bad.backend_fit.posterior = object()
    with pytest.raises(GP3BayesError):
        po._posterior_data_vars(bad)

    bad = _fake_fit()
    bad.backend_fit.posterior = {"x": _arr(np.ones(3))}
    with pytest.raises(GP3BayesError):
        po._posterior_components(bad)

    bad = _fake_fit()
    bad.backend_fit.posterior = {}
    with pytest.raises(GP3BayesError):
        po._posterior_components(bad)

    assert np.isnan(po._dataset_scalar({}, "x"))
    assert np.isnan(po._dataset_scalar({"x": _arr(np.array([]))}, "x"))

    assert po._classify_upper(np.nan, 1, 2) == "not_assessed"
    assert po._classify_upper(0.5, 1, 2) == "pass"
    assert po._classify_upper(1.5, 1, 2) == "review"
    assert po._classify_upper(3, 1, 2) == "fail"
    assert po._classify_lower(np.nan, 2, 1) == "not_assessed"
    assert po._classify_lower(3, 2, 1) == "pass"
    assert po._classify_lower(1.5, 2, 1) == "review"
    assert po._classify_lower(0.5, 2, 1) == "fail"
    assert po._worst_status([]) == "review"
    assert po._worst_status(["pass"]) == "pass"
    assert po._worst_status(["pass", "not_applicable"]) == "review"
    assert po._worst_status(["fail", "pass"]) == "fail"

    assert po._sample_stats_array(object(), ("x",)) is None
    assert po._sample_stats_array({"x": _arr(np.ones((2, 3)))}, ("x",)).shape == (2, 3)
    assert po._sample_stats_array({"x": _arr(np.ones(3))}, ("x",)) is None


def test_prior_bridge_sampling_guards_tables_long_and_plots():
    spec = _prior_spec()
    priors = spec.priors
    assert pb._prior_object(priors) is priors
    assert pb._prior_object(spec) is priors
    with pytest.raises(GP3BayesError):
        pb._prior_object(object())

    assert pb._class_for_variable("b_Intercept") == "Intercept"
    assert pb._class_for_variable("b_x") == "b"
    assert pb._class_for_variable("sd_p") == "sd"
    assert pb._class_for_variable("cor_p") == "cor"
    assert pb._class_for_variable("sigma") == "sigma"
    assert pb._class_for_variable("unknown") is None

    rng = np.random.default_rng(1)
    normal = pd.Series(
        {
            "distribution": "normal",
            "location": 0,
            "scale": 1,
            "df": np.nan,
            "lower": np.nan,
            "shape": np.nan,
        }
    )
    student = pd.Series(
        {
            "distribution": "student_t",
            "location": 0,
            "scale": 1,
            "df": 3,
            "lower": 0,
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
            "shape": 2,
        }
    )
    assert pb._sample_prior(normal, 10, rng).shape == (10,)
    assert np.all(pb._sample_prior(student, 10, rng) >= 0)
    assert np.all(np.abs(pb._sample_prior(lkj, 10, rng)) <= 1)
    bad = normal.copy()
    bad["distribution"] = "bad"
    with pytest.raises(GP3BayesError):
        pb._sample_prior(bad, 10, rng)

    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            priors,
            variables="b_Intercept",
            ndraws=10,
        )
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(priors, variables=None)
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            priors,
            variables=("b_Intercept",),
            regex="no-match",
            ndraws=50,
        )
    with pytest.raises(GP3BayesError):
        pb.simulate_declared_prior_draws(
            priors,
            variables=("unknown",),
            ndraws=50,
        )
    draws = pb.simulate_declared_prior_draws(
        priors,
        variables=("b_Intercept", "b_x"),
        ndraws=50,
        seed=2,
    )
    assert draws.shape == (50, 2)

    assert np.isnan(pb._overlap(1, 1, 1, 1))
    summary = pb._summary(np.arange(5.0), (0.1, 0.5, 0.9))
    assert "median" in summary

    prior = pd.DataFrame({"b_Intercept": np.linspace(-1, 1, 80)})
    posterior = pd.DataFrame({"b_Intercept": np.linspace(-0.5, 0.5, 80)})
    summary_table = pd.DataFrame(
        {
            "variable": ["b_Intercept"],
            "prior_lower": [-1.0],
            "prior_median": [0.0],
            "prior_upper": [1.0],
            "posterior_lower": [-0.5],
            "posterior_median": [0.0],
            "posterior_upper": [0.5],
            "prior_sd": [0.5],
            "posterior_sd": [0.25],
            "median_shift": [0.0],
            "standardized_location_shift": [0.0],
            "sd_ratio": [0.5],
            "contraction": [0.5],
            "interval_overlap_fraction": [0.5],
        }
    )
    distances = pd.DataFrame(
        {
            "variable": ["b_Intercept"],
            "ks_distance": [0.1],
            "quantile_wasserstein": [0.2],
            "standardized_quantile_wasserstein": [0.4],
        }
    )
    bridge = pb.PriorPosteriorBridge(
        "0.3",
        "binary",
        ("b_Intercept",),
        prior,
        posterior,
        summary_table,
        distances,
        (0.025, 0.5, 0.975),
        1,
    )
    assert bridge.to_frame().equals(summary_table)
    assert pb.prior_posterior_summary_table(bridge).equals(summary_table)
    assert pb.prior_posterior_distance_table(bridge).equals(distances)
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_summary_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        pb.prior_posterior_draws_long(bridge, max_draws=10)
    long = pb.prior_posterior_draws_long(bridge, max_draws=50, seed=4)
    assert set(long["distribution"]) == {"prior", "posterior"}
    assert pb.plot_prior_posterior_density(bridge, max_draws=50).axes
    assert pb.plot_prior_posterior_intervals(bridge).axes
