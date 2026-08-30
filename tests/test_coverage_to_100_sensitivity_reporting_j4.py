from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.reporting as rep
import gp3bayespy.sensitivity as s
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_sensitivity_core_helpers_plan_empty_suite_and_evidence(tmp_path):
    assert s._family(SimpleNamespace(family="binary")) == "binary"
    with pytest.raises(GP3BayesError):
        s._family(SimpleNamespace(family="pupil"))
    assert s._flag(np.bool_(True), "x") is True
    with pytest.raises(GP3BayesError):
        s._flag(1, "x")
    assert s._mapping(None, "x") == {}
    assert s._mapping({"a": 1}, "x") == {"a": 1}
    with pytest.raises(GP3BayesError):
        s._mapping([], "x")  # type: ignore[arg-type]

    frame = pd.DataFrame({"x": [1]})
    assert s._table_field(SimpleNamespace(table=frame), "table").equals(frame)
    assert s._table_field({"table": frame}, "table").equals(frame)
    with pytest.raises(GP3BayesError):
        s._table_field(object(), "table")

    assert s._status(None) == "not_run"
    assert s._status(s.SuiteError("error", "x")) == "error"
    assert s._status({"status": "warn"}) == "warn"
    assert s._status(SimpleNamespace(adequacy_established=True)) == "pass"
    assert s._status(object()) == "completed"

    assert s._worst_status([]) == "not_assessed"
    assert s._worst_status(["pass", "error"]) == "error"
    assert s._classify_upper(np.nan, 1, 2) == "review"
    assert s._classify_upper(0.5, 1, 2) == "pass"
    assert s._classify_upper(1.5, 1, 2) == "review"
    assert s._classify_upper(3, 1, 2) == "fail"

    with pytest.raises(GP3BayesError):
        s.create_sensitivity_suite_plan(prior_scale=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        s.create_sensitivity_suite_plan(prior_scale_args=[])  # type: ignore[arg-type]

    plan = s.create_sensitivity_suite_plan()
    fit = SimpleNamespace(family="binary")
    suite = s.run_sensitivity_suite(fit, plan)
    assert suite.status == "not_run"
    assert s.summarise_sensitivity_suite(suite).empty

    with pytest.raises(GP3BayesError):
        s.run_sensitivity_suite(fit, object())  # type: ignore[arg-type]

    assert s._safe_call(lambda x=1: x, {}, False) == 1
    error = s._safe_call(
        lambda: (_ for _ in ()).throw(RuntimeError("x")),
        {},
        False,
    )
    assert isinstance(error, s.SuiteError)
    with pytest.raises(RuntimeError):
        s._safe_call(
            lambda: (_ for _ in ()).throw(RuntimeError("x")),
            {},
            True,
        )

    err_suite = s.SensitivitySuite(
        "0.2",
        "binary",
        "review",
        fit,
        plan,
        None,
        {"x": s.SuiteError("error", "detail")},
        pd.DataFrame({"component": ["x"], "status": ["error"]}),
    )
    summary = s.summarise_sensitivity_suite(err_suite)
    assert summary.iloc[0]["detail"] == "detail"
    with pytest.raises(GP3BayesError):
        s.summarise_sensitivity_suite(object())  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        s.collect_model_evidence(compute=("bad",))
    with pytest.raises(GP3BayesError):
        s.collect_model_evidence(compute=("diagnostics",))

    evidence = s.collect_model_evidence(
        design={"status": "pass"},
        diagnostics=SimpleNamespace(status="review"),
        estimands=SimpleNamespace(family="duration", status="completed"),
    )
    assert evidence.family == "duration"
    path = tmp_path / "evidence.md"
    returned = s.create_model_evidence_report(evidence, path)
    assert returned == str(path.resolve())
    assert "model evidence report" in path.read_text(encoding="utf-8")
    with pytest.raises(GP3BayesError):
        s.create_model_evidence_report(evidence, path)
    with pytest.raises(GP3BayesError):
        s.create_model_evidence_report(
            evidence,
            tmp_path / "missing" / "evidence.md",
        )
    with pytest.raises(GP3BayesError):
        s.create_model_evidence_report(object(), path)  # type: ignore[arg-type]


def test_sensitivity_publication_adapters_and_plots():
    parameter = pd.DataFrame(
        {
            "variable": ["b_x"],
            "standardized_bias": [0.1],
            "coverage": [0.95],
            "rmse": [0.2],
        }
    )
    estimates = pd.DataFrame(
        {
            "variable": ["b_x", "b_x"],
            "truth": [0.2, 0.2],
            "median": [0.1, 0.3],
            "lower": [0.0, 0.1],
            "upper": [0.2, 0.5],
            "repetition": [1, 2],
        }
    )
    status = pd.DataFrame(
        {
            "diagnostic_status": ["pass", "review"],
            "completed": [True, False],
        }
    )
    recovery = SimpleNamespace(
        parameter_summary=parameter,
        estimates=estimates,
        fit_status=status,
    )
    assert s.recovery_parameter_table(recovery).equals(parameter)
    assert s.recovery_estimate_table(recovery).equals(estimates)
    assert s.recovery_fit_status_table(recovery).equals(status)
    assert s.plot_recovery_bias(recovery).axes
    assert s.plot_recovery_coverage(recovery).axes
    assert s.plot_recovery_rmse(recovery).axes
    assert s.plot_recovery_estimates(recovery).axes
    with pytest.raises(GP3BayesError):
        s.plot_recovery_estimates(recovery, variables=("missing",))
    assert s.plot_recovery_fit_status(recovery).axes

    comparison = pd.DataFrame({"scenario": ["a"], "value": [1.0]})
    scenario = pd.DataFrame({"scenario": ["a"], "status": ["pass"]})
    prior = SimpleNamespace(
        comparison=comparison,
        scenario_status=scenario,
    )
    assert s.prior_sensitivity_table(prior).equals(comparison)
    assert s.prior_sensitivity_scenario_table(prior).equals(scenario)
    assert s.estimand_sensitivity_table({"table": comparison}).equals(comparison)
    assert s.group_deletion_sensitivity_table({"summary": comparison}).equals(comparison)

    with pytest.raises(GP3BayesError):
        s.random_slope_sensitivity_table({})
    assert s.random_slope_sensitivity_table({"comparison": {"table": comparison}}).equals(
        comparison
    )

    assert s.powerscale_sensitivity_table(comparison).equals(comparison)
    assert s.powerscale_sensitivity_table({"raw": [{"x": 1}]}).iloc[0]["x"] == 1

    raw_sbc = {"raw": {"stats": [{"variable": "b", "rank": 1}]}}
    assert not s.sbc_stats_table(raw_sbc).empty
    overview = s.sbc_overview_table(
        {
            "raw": {"stats": [{"variable": "b", "rank": 1}]},
            "plan": {
                "family": "binary",
                "backend": "none",
                "n_sims": 1,
            },
            "status": "completed",
            "diagnostics_inspected": True,
        }
    )
    assert overview.iloc[0]["family"] == "binary"
    with pytest.raises(GP3BayesError):
        s.sbc_stats_table({})


def test_reporting_core_data_adapters_dashboard_and_interval_plots(tmp_path):
    assert rep.theme_gp3bayes(10)["base_size"] == 10
    with pytest.raises(GP3BayesError):
        rep.theme_gp3bayes(0)

    frame = pd.DataFrame({"x": [1]})
    assert rep._df(frame, "table").equals(frame)
    assert rep._df(SimpleNamespace(table=frame), "table").equals(frame)
    assert rep._df({"table": frame}, "table").equals(frame)
    with pytest.raises(GP3BayesError):
        rep._df(object(), "table")

    assert rep._status(None) == "not_available"
    assert rep._status(object()) == "available"
    assert rep._status({"status": "review"}) == "review"

    arr = np.arange(20.0).reshape(10, 2)
    posterior = rep._posterior_frame(arr)
    assert len(posterior) == 2
    direct = pd.DataFrame(
        {
            "variable": ["b"],
            "lower": [0.0],
            "median": [1.0],
            "upper": [2.0],
        }
    )
    assert rep._posterior_frame(direct).equals(direct)

    with pytest.raises(GP3BayesError):
        rep.create_complete_evidence_inventory({})
    inventory = rep.create_complete_evidence_inventory(
        {"a": None, "b": SimpleNamespace(status="pass")},
        label="x",
    )
    assert len(rep.evidence_inventory_table(inventory)) == 2
    with pytest.raises(GP3BayesError):
        rep.evidence_inventory_table(object())  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        rep.create_diagnostic_dashboard()
    dashboard = rep.create_diagnostic_dashboard(
        analysis_bundle=SimpleNamespace(status="pass"),
        label="dash",
    )
    assert not rep.diagnostic_dashboard_table(dashboard).empty
    assert rep.plot_diagnostic_dashboard(dashboard).axes
    with pytest.raises(GP3BayesError):
        rep.create_diagnostic_dashboard_figures(dashboard)

    report = tmp_path / "dashboard.md"
    returned = rep.write_diagnostic_dashboard_report(dashboard, report)
    assert returned == str(report.resolve())
    with pytest.raises(GP3BayesError):
        rep.write_diagnostic_dashboard_report(dashboard, report)

    pred = pd.DataFrame(
        {
            "predicted_mean": [0.2, 0.8],
            "lower": [0.1, 0.7],
            "upper": [0.3, 0.9],
        }
    )
    assert rep.plot_prediction_intervals(pred).axes
    assert rep.plot_prediction_intervals(pd.DataFrame({"mean": [1.0]})).axes

    calibration = pd.DataFrame(
        {
            "mean_predicted_probability": [0.2, 0.8],
            "observed_rate": [0.1, 0.9],
        }
    )
    assert rep.plot_binary_calibration(calibration).axes

    thresholds = pd.DataFrame(
        {
            "threshold": [0.5],
            "accuracy": [0.8],
            "sensitivity": [0.7],
            "specificity": [0.9],
            "balanced_accuracy": [0.8],
        }
    )
    assert rep.plot_binary_threshold_metrics(thresholds).axes
