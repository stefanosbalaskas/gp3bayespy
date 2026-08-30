from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import gp3bayespy.sensitivity as s
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_remaining_sensitivity_plot_and_adapter_guards():
    bad = pd.DataFrame({"x": [1]})
    for func in (
        s.plot_recovery_bias,
        s.plot_recovery_coverage,
        s.plot_recovery_rmse,
        s.plot_recovery_estimates,
        s.plot_recovery_fit_status,
        s.plot_prior_sensitivity,
        s.plot_prior_sensitivity_scenarios,
        s.plot_estimand_sensitivity_gg,
        s.plot_group_deletion_sensitivity,
        s.plot_powerscale_sensitivity_gg,
    ):
        with pytest.raises(GP3BayesError):
            func(bad)

    prior = pd.DataFrame(
        {
            "scenario": ["a", "b"],
            "scale_multiplier": [0.5, 1.5],
            "variable": ["b", "b"],
            "standardized_shift": [0.1, 0.2],
        }
    )
    assert s.plot_prior_sensitivity(prior).axes

    scenarios = pd.DataFrame(
        {
            "scenario": ["a", "b"],
            "maximum_standardized_shift": [0.1, 0.2],
        }
    )
    assert s.plot_prior_sensitivity_scenarios(scenarios).axes

    estimand = pd.DataFrame(
        {
            "alternative": ["a"],
            "reference_median": [1.0],
            "alternative_median": [1.1],
            "alternative_lower": [0.9],
            "alternative_upper": [1.3],
        }
    )
    assert s.plot_estimand_sensitivity_gg(estimand).axes
    assert s.plot_random_slope_sensitivity(estimand).axes

    deletion = pd.DataFrame({"omitted_unit": ["p1"], "median_shift": [0.1]})
    assert s.plot_group_deletion_sensitivity(deletion).axes

    powerscale1 = pd.DataFrame({"variable": ["b"], "prior": [0.1], "likelihood": [0.2]})
    assert s.plot_powerscale_sensitivity_gg(powerscale1).axes

    powerscale2 = pd.DataFrame(
        {
            "variable": ["b"],
            "component": ["prior"],
            "alpha": [1.0],
            "distance": [0.1],
        }
    )
    assert s.plot_powerscale_sensitivity_gg(powerscale2).axes

    with pytest.raises(GP3BayesError):
        s.powerscale_sensitivity_table(object())

    sbc = {
        "raw": {
            "stats": [
                {"variable": "b", "rank": 1, "draws": 10},
                {"variable": "b", "rank": 5, "draws": 10},
            ]
        }
    }
    assert s.plot_sbc_rank_gg(sbc).axes
    assert s.plot_sbc_ecdf_gg(sbc).axes
    assert s.plot_sbc_coverage_gg(sbc).axes
    with pytest.raises(GP3BayesError):
        s.plot_sbc_rank_gg(sbc, variables=("missing",))

    coverage = {"raw": {"stats": [{"variable": "b", "rank": 1, "draws": 10, "coverage": 0.9}]}}
    assert s.plot_sbc_coverage_gg(coverage).axes


def test_collect_model_evidence_computed_components(monkeypatch):
    fit = SimpleNamespace(family="binary")
    import gp3bayespy.unified_workflow_api as u

    monkeypatch.setattr(u, "diagnose_model_fit", lambda fit: SimpleNamespace(status="pass"))
    monkeypatch.setattr(
        u,
        "summarise_model_posterior",
        lambda fit: SimpleNamespace(status="available"),
    )
    monkeypatch.setattr(
        u,
        "estimate_model_estimands",
        lambda fit: SimpleNamespace(family="binary", status="completed"),
    )
    evidence = s.collect_model_evidence(
        fit=fit,
        compute=("diagnostics", "posterior", "estimands"),
    )
    supplied = evidence.component_table[evidence.component_table["available"]]
    assert len(supplied) == 3
