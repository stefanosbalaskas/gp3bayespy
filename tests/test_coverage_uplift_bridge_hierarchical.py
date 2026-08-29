from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import importlib

import gp3bayespy.hierarchical_effects_advanced as hea
import gp3bayespy.posterior_validation_core as pvc
from gp3bayespy.exceptions import GP3BayesError
from gp3bayespy.specification import PriorSpecification

ppb = importlib.import_module("gp3bayespy.prior_posterior_bridge")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class DV:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


def _prior() -> PriorSpecification:
    table = pd.DataFrame(
        [
            [
                "Intercept",
                "normal",
                "Intercept",
                0.0,
                1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "baseline",
            ],
            ["b", "normal", "b", 0.0, 0.5, np.nan, np.nan, np.nan, np.nan, "coefficient"],
            ["sd", "student_t", "sd", 0.0, 0.5, 3.0, np.nan, 0.0, np.nan, "group sd"],
            ["cor", "lkj", "cor", np.nan, np.nan, np.nan, 2.0, np.nan, np.nan, "correlation"],
            ["sigma", "student_t", "sigma", 0.0, 0.4, 3.0, np.nan, 0.0, np.nan, "residual"],
        ],
        columns=[
            "parameter_class",
            "distribution",
            "target",
            "location",
            "scale",
            "df",
            "shape",
            "lower",
            "upper",
            "rationale",
        ],
    )
    return PriorSpecification("0.2", "binary", "Bernoulli", None, True, 0.5, 0.0, table)


def test_prior_simulation_bridge_tables_and_all_plots(monkeypatch):
    prior = _prior()
    requested = ["b_Intercept", "b_condition", "sd_participant", "cor_participant", "sigma"]
    draws = ppb.simulate_declared_prior_draws(prior, variables=requested, ndraws=200, seed=7)
    assert list(draws) == requested
    assert len(ppb.prior_specification_table(prior)) == 5

    posterior = pd.DataFrame(
        {
            "b_Intercept": np.linspace(-0.2, 0.2, 180),
            "b_condition": np.linspace(0.1, 0.8, 180),
            "sd_participant": np.linspace(0.05, 0.55, 180),
            "cor_participant": np.linspace(-0.5, 0.5, 180),
            "sigma": np.linspace(0.15, 0.45, 180),
        }
    )
    fit = SimpleNamespace(
        family="binary",
        fit_performed=True,
        specification=SimpleNamespace(priors=prior),
    )
    monkeypatch.setattr(ppb, "extract_posterior_draws", lambda *args, **kwargs: posterior.copy())
    bridge = ppb.prior_posterior_bridge(fit, ndraws=150, seed=9)
    assert len(bridge.to_frame()) == 5
    assert len(ppb.prior_posterior_summary_table(bridge)) == 5
    assert len(ppb.prior_posterior_distance_table(bridge)) == 5
    assert set(ppb.prior_posterior_draws_long(bridge, max_draws=100)["distribution"]) == {
        "prior",
        "posterior",
    }
    figures = (
        ppb.plot_prior_posterior_density(bridge, max_draws=100),
        ppb.plot_prior_posterior_intervals(bridge),
        ppb.plot_prior_posterior_shift(bridge),
        ppb.plot_prior_posterior_contraction(bridge),
    )
    assert all(fig.axes for fig in figures)

    with pytest.raises(GP3BayesError, match="variables"):
        ppb.simulate_declared_prior_draws(prior, ndraws=100)
    with pytest.raises(GP3BayesError, match="ndraws"):
        ppb.simulate_declared_prior_draws(prior, variables="b_Intercept", ndraws=10)
    with pytest.raises(GP3BayesError, match="Unsupported variable"):
        ppb.simulate_declared_prior_draws(prior, variables="unknown", ndraws=100)
    with pytest.raises(GP3BayesError, match="PriorPosteriorBridge"):
        ppb.prior_posterior_summary_table(object())  # type: ignore[arg-type]


def _fake_hier_fit(family="binary"):
    contract = SimpleNamespace(
        mappings={"participant": "participant_id", "item": "item_id", "condition": "condition"}
    )
    data = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p1", "p2"],
            "item_id": ["i1", "i2", "i3", "i1", "i2", "i3"],
            "condition": ["A", "B", "A", "B", "A", "B"],
        }
    )
    return SimpleNamespace(
        family=family,
        specification=SimpleNamespace(prepared=SimpleNamespace(data=data, contract=contract)),
        fit_performed=True,
    )


def test_hierarchical_group_effects_variance_partitions_and_plots(monkeypatch):
    fit = _fake_hier_fit("binary")
    monkeypatch.setattr(hea, "_validate_fit_like", lambda x: x)

    participant_z = np.arange(2 * 6 * 2 * 2, dtype=float).reshape(2, 6, 2, 2) / 20
    participant_sd = np.ones((2, 6, 2)) * np.array([0.3, 0.2])
    item_z = np.arange(2 * 6 * 3, dtype=float).reshape(2, 6, 3) / 20
    item_sd = np.ones((2, 6)) * 0.25
    variables = {
        "participant_chol_stds": DV(participant_sd),
        "participant_z": DV(participant_z),
        "sd_item": DV(item_sd),
        "item_z": DV(item_z),
    }
    monkeypatch.setattr(hea, "_posterior_data_vars", lambda x: variables)

    table = hea.group_effect_draws_table(fit, ndraws=5, seed=3)
    assert {"group", "level", "coefficient", "draw", "value"} <= set(table)
    ranks = hea.group_effect_rank_probability_table(
        fit, "participant_id", coefficient="Intercept", ndraws=6
    )
    assert np.isclose(ranks["probability_highest"].sum(), 1.0)
    assert not ranks["automatic_ranking_decision"].any()

    fig1 = hea.plot_group_effect_distribution(table, max_levels=5)
    fig2 = hea.plot_group_effect_rank_probability(ranks)
    assert fig1.axes and fig2.axes

    components_binary = {
        "sd_participant__Intercept": np.full((2, 6), 0.4),
        "sd_item__Intercept": np.full((2, 6), 0.2),
    }
    monkeypatch.setattr(hea, "_posterior_components", lambda x: components_binary)
    partition = hea.random_intercept_variance_partition(fit)
    assert len(hea.random_intercept_variance_partition_table(partition)) == 3
    assert hea.plot_random_intercept_variance_partition(partition).axes

    duration_fit = _fake_hier_fit("duration")
    components_duration = {
        "sd_participant__Intercept": np.full((2, 6), 0.3),
        "sigma": np.full((2, 6), 0.7),
    }
    monkeypatch.setattr(hea, "_posterior_components", lambda x: components_duration)
    duration_partition = hea.random_intercept_variance_partition(duration_fit)
    assert set(duration_partition.table["component_type"]) == {"random_intercept", "residual"}

    with pytest.raises(GP3BayesError, match="Unknown grouping"):
        hea.group_effect_draws_table(fit, groups="missing")
    with pytest.raises(GP3BayesError, match="max_rows"):
        hea.group_effect_draws_table(fit, max_rows=2)
    with pytest.raises(GP3BayesError, match="probs"):
        hea.random_intercept_variance_partition(duration_fit, probs=(0, 0.5, 0.9))


def test_posterior_validation_trace_energy_treedepth_divergence_and_errors(monkeypatch):
    fit = object()
    monkeypatch.setattr(pvc, "_validate_fit_like", lambda x: x)
    components = {
        "b_0": np.array([[0.1, 0.2, 0.0, 0.3], [0.0, 0.1, 0.2, 0.1]]),
        "b_1": np.array([[0.4, 0.5, 0.6, 0.5], [0.3, 0.4, 0.5, 0.6]]),
    }
    monkeypatch.setattr(pvc, "_posterior_components", lambda x: components)
    sampler = pd.DataFrame(
        {
            "Chain": np.tile([1, 2], 12),
            "Iteration": np.repeat(np.arange(1, 13), 2),
            "Parameter": np.tile(["energy", "tree_depth", "diverging"], 8),
            "Value": np.tile([3.0, 8.0, 0.0, 3.2, 9.0, 1.0], 4),
        }
    )
    monkeypatch.setattr(pvc, "extract_sampler_diagnostics", lambda x: sampler)

    figures = (
        pvc.plot_sampling_diagnostics(fit, "trace"),
        pvc.plot_sampling_diagnostics(fit, "trace", variables=["b_1"]),
        pvc.plot_sampling_diagnostics(fit, "energy"),
        pvc.plot_sampling_diagnostics(fit, "treedepth"),
        pvc.plot_sampling_diagnostics(fit, "divergence"),
    )
    assert all(fig.axes for fig in figures)

    with pytest.raises(GP3BayesError, match="one of"):
        pvc.plot_sampling_diagnostics(fit, "bad")
    with pytest.raises(GP3BayesError, match="not found"):
        pvc.plot_sampling_diagnostics(fit, "trace", variables=["missing"])
