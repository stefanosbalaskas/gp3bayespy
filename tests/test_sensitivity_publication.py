from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gp3bayespy.sensitivity import (
    collect_model_evidence,
    create_model_evidence_report,
    create_sensitivity_suite_plan,
    plot_recovery_bias,
    plot_recovery_coverage,
    plot_recovery_rmse,
    prior_sensitivity_scenario_table,
    prior_sensitivity_table,
    recovery_estimate_table,
    recovery_fit_status_table,
    recovery_parameter_table,
    run_sensitivity_suite,
    sbc_overview_table,
    sbc_stats_table,
    summarise_sensitivity_suite,
)


@dataclass
class FakeFit:
    family: str = "binary"


def test_plan_is_inert_and_conservative():
    plan = create_sensitivity_suite_plan()
    assert plan.prior_scale["run"] is False
    assert plan.powerscale["run"] is False
    assert plan.psis_loo["run"] is False
    assert plan.automatic_model_selection is False
    assert plan.automatic_exclusion is False


def test_empty_suite_performs_no_work():
    suite = run_sensitivity_suite(FakeFit())
    assert suite.status == "not_run"
    assert suite.results == {}
    assert suite.robustness_established is False
    assert suite.automatic_model_selection is False
    assert summarise_sensitivity_suite(suite).empty


def test_evidence_inventory_and_report(tmp_path: Path):
    design = {"status": "pass"}
    evidence = collect_model_evidence(fit=FakeFit(), design=design)
    row = evidence.component_table.loc[evidence.component_table["component"] == "design"].iloc[0]
    assert bool(row["available"])
    assert evidence.adequacy_established is False
    assert evidence.robustness_established is False
    assert evidence.causal_identification_established is False
    target = tmp_path / "evidence.md"
    result = create_model_evidence_report(evidence, target)
    assert Path(result).exists()
    assert "evidence inventory" in target.read_text(encoding="utf-8").lower()


def test_recovery_and_sensitivity_adapters():
    recovery = {
        "parameter_summary": pd.DataFrame(
            {
                "variable": ["a", "b"],
                "standardized_bias": [0.1, -0.2],
                "coverage": [0.9, 0.85],
                "rmse": [0.2, 0.3],
            }
        ),
        "estimates": pd.DataFrame(
            {
                "variable": ["a", "a"],
                "truth": [0, 0],
                "median": [0.1, -0.1],
                "lower": [-0.2, -0.3],
                "upper": [0.3, 0.2],
                "repetition": [1, 2],
            }
        ),
        "fit_status": pd.DataFrame(
            {"repetition": [1, 2], "diagnostic_status": ["pass", "pass"], "completed": [True, True]}
        ),
    }
    assert len(recovery_parameter_table(recovery)) == 2
    assert len(recovery_estimate_table(recovery)) == 2
    assert len(recovery_fit_status_table(recovery)) == 2
    sensitivity = {
        "comparison": pd.DataFrame(
            {
                "scenario": ["tight", "wide"],
                "scale_multiplier": [0.5, 2.0],
                "variable": ["b_x", "b_x"],
                "standardized_shift": [0.1, 0.2],
            }
        ),
        "scenario_status": pd.DataFrame(
            {"scenario": ["tight", "wide"], "maximum_standardized_shift": [0.1, 0.2]}
        ),
    }
    assert len(prior_sensitivity_table(sensitivity)) == 2
    assert len(prior_sensitivity_scenario_table(sensitivity)) == 2


def test_recovery_plots_are_real_figures():
    d = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "standardized_bias": [0.1, -0.1],
            "coverage": [0.9, 0.85],
            "rmse": [0.2, 0.3],
        }
    )
    for fig in (plot_recovery_bias(d), plot_recovery_coverage(d), plot_recovery_rmse(d)):
        assert hasattr(fig, "savefig")


def test_sbc_adapters_preserve_no_calibration_claim():
    x = {
        "status": "review",
        "plan": {"family": "binary", "backend": "example", "n_sims": 50},
        "raw": {
            "stats": pd.DataFrame({"variable": ["a", "b"], "rank": [1, 2], "max_rank": [100, 100]})
        },
        "diagnostics_inspected": True,
        "calibration_established": False,
    }
    assert len(sbc_stats_table(x)) == 2
    overview = sbc_overview_table(x)
    assert int(overview.loc[0, "simulations"]) == 50
    assert bool(overview.loc[0, "calibration_established"]) is False
