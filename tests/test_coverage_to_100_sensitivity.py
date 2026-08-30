from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.advanced_optional_workflows as aow
import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.sensitivity as s
import gp3bayespy.specification_closure as closure
import gp3bayespy.unified_workflow_api as unified
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_sensitivity_helpers_status_mapping_classification_and_safe_call():
    assert s._family(SimpleNamespace(family="binary")) == "binary"
    assert s._family(SimpleNamespace(family="duration")) == "duration"
    with pytest.raises(GP3BayesError):
        s._family(SimpleNamespace(family="other"))

    assert s._flag(True, "x") is True
    assert s._flag(np.bool_(False), "x") is False
    with pytest.raises(GP3BayesError):
        s._flag(1, "x")

    assert s._mapping(None, "x") == {}
    assert s._mapping({"a": 1}, "x") == {"a": 1}
    with pytest.raises(GP3BayesError):
        s._mapping([("a", 1)], "x")  # type: ignore[arg-type]

    frame = pd.DataFrame({"x": [1]})
    assert s._table_field(SimpleNamespace(table=frame), "table").equals(frame)
    assert s._table_field({"table": frame}, "table").equals(frame)
    with pytest.raises(GP3BayesError):
        s._table_field(object(), "table")

    assert s._status(None) == "not_run"
    assert s._status(s.SuiteError("error", "bad")) == "error"
    assert s._status({"status": "review"}) == "review"
    assert s._status(SimpleNamespace(status="pass")) == "pass"
    assert s._status(SimpleNamespace(adequacy_established=True)) == "pass"
    assert s._status(object()) == "completed"

    assert s._worst_status([]) == "not_assessed"
    assert s._worst_status(["pass", "completed"]) == "pass"
    assert s._worst_status(["pass", "review"]) == "review"
    assert s._worst_status(["pass", "fail"]) == "fail"
    assert s._worst_status(["error"]) == "error"

    assert s._classify_upper(np.nan, 0.2, 0.5) == "review"
    assert s._classify_upper(0.1, 0.2, 0.5) == "pass"
    assert s._classify_upper(0.3, 0.2, 0.5) == "review"
    assert s._classify_upper(0.8, 0.2, 0.5) == "fail"

    assert s._safe_call(lambda x=1: x + 1, {"x": 3}, False) == 4
    err = s._safe_call(lambda: 1 / 0, {}, False)
    assert isinstance(err, s.SuiteError)
    with pytest.raises(ZeroDivisionError):
        s._safe_call(lambda: 1 / 0, {}, True)


def test_sensitivity_plan_validation_and_empty_suite():
    plan = s.create_sensitivity_suite_plan(
        prior_scale=True,
        powerscale=True,
        psis_loo=True,
        prior_scale_args={"a": 1},
        powerscale_args={"b": 2},
        psis_args={"c": 3},
        alternative_estimands={"alt": object()},
    )
    assert plan.prior_scale["run"]
    assert plan.powerscale["run"]
    assert plan.psis_loo["run"]
    assert "alt" in plan.alternative_estimands

    for kwargs in (
        {"prior_scale": 1},
        {"powerscale": "yes"},
        {"psis_loo": None},
        {"prior_scale_args": []},
        {"alternative_estimands": []},
    ):
        with pytest.raises(GP3BayesError):
            s.create_sensitivity_suite_plan(**kwargs)  # type: ignore[arg-type]

    fit = SimpleNamespace(family="binary")
    empty = s.run_sensitivity_suite(fit)
    assert empty.status == "not_run"
    assert s.summarise_sensitivity_suite(empty).empty

    with pytest.raises(GP3BayesError):
        s.run_sensitivity_suite(SimpleNamespace(family="other"))
    with pytest.raises(GP3BayesError):
        s.run_sensitivity_suite(fit, plan=object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        s.run_sensitivity_suite(fit, stop_on_error=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        s.summarise_sensitivity_suite(object())  # type: ignore[arg-type]


def test_run_sensitivity_suite_all_enabled_components(monkeypatch):
    result = SimpleNamespace(status="pass", interpretation="inspected")
    monkeypatch.setattr(binary, "assess_binary_prior_sensitivity", lambda **kwargs: result)
    monkeypatch.setattr(duration, "assess_duration_prior_sensitivity", lambda **kwargs: result)
    monkeypatch.setattr(aow, "assess_powerscaled_sensitivity", lambda **kwargs: result)
    monkeypatch.setattr(aow, "compute_psis_loo", lambda **kwargs: result)
    monkeypatch.setattr(closure, "run_random_slope_sensitivity", lambda **kwargs: result)
    monkeypatch.setattr(closure, "run_group_deletion_sensitivity", lambda **kwargs: result)
    monkeypatch.setattr(closure, "compare_estimand_sensitivity", lambda **kwargs: result)
    monkeypatch.setattr(closure, "audit_duration_unit_invariance", lambda **kwargs: result)

    bfit = SimpleNamespace(family="binary")
    plan = s.create_sensitivity_suite_plan(
        prior_scale=True,
        powerscale=True,
        psis_loo=True,
        random_slope_plan=object(),
        group_deletion_plan=object(),
        alternative_estimands={"alt": object()},
    )
    suite = s.run_sensitivity_suite(
        bfit,
        plan=plan,
        reference_estimand=object(),
    )
    assert suite.status == "completed"
    assert set(suite.results) == {
        "prior_scale",
        "powerscale",
        "psis_loo",
        "random_slope",
        "group_deletion",
        "estimand_alternatives",
    }
    summary = s.summarise_sensitivity_suite(suite)
    assert len(summary) == 6
    assert summary["status"].eq("pass").all()

    dfit = SimpleNamespace(family="duration")
    dplan = s.create_sensitivity_suite_plan(
        prior_scale=True,
        duration_unit={"estimand": object(), "multiplier": 1000.0},
    )
    dsuite = s.run_sensitivity_suite(
        dfit,
        plan=dplan,
        reference_estimand=object(),
    )
    assert set(dsuite.results) == {"prior_scale", "duration_unit"}

    with pytest.raises(GP3BayesError):
        s.run_sensitivity_suite(
            dfit,
            plan=s.create_sensitivity_suite_plan(duration_unit={"multiplier": 2}),
            reference_estimand=object(),
        )


def test_sensitivity_suite_error_paths_and_auto_reference(monkeypatch):
    monkeypatch.setattr(
        aow,
        "compute_psis_loo",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("loo failed")),
    )
    plan = s.create_sensitivity_suite_plan(psis_loo=True)
    suite = s.run_sensitivity_suite(SimpleNamespace(family="binary"), plan=plan)
    assert suite.status == "review"
    assert isinstance(suite.results["psis_loo"], s.SuiteError)
    summary = s.summarise_sensitivity_suite(suite)
    assert "loo failed" in summary.loc[0, "detail"]

    with pytest.raises(RuntimeError, match="loo failed"):
        s.run_sensitivity_suite(
            SimpleNamespace(family="binary"),
            plan=plan,
            stop_on_error=True,
        )

    monkeypatch.setattr(
        unified,
        "estimate_model_estimands",
        lambda fit: (_ for _ in ()).throw(RuntimeError("estimand failed")),
    )
    auto_plan = s.create_sensitivity_suite_plan(alternative_estimands={"alt": object()})
    auto = s.run_sensitivity_suite(SimpleNamespace(family="binary"), plan=auto_plan)
    assert auto.reference_estimand.status == "error"
    assert "estimand_alternatives" not in auto.results

    monkeypatch.setattr(unified, "estimate_model_estimands", lambda fit: object())
    monkeypatch.setattr(
        closure,
        "compare_estimand_sensitivity",
        lambda **kwargs: SimpleNamespace(status="review", interpretation="inspect"),
    )
    auto2 = s.run_sensitivity_suite(SimpleNamespace(family="binary"), plan=auto_plan)
    assert "estimand_alternatives" in auto2.results
    assert auto2.status == "review"


def test_collect_evidence_compute_inventory_and_report(monkeypatch, tmp_path):
    component = SimpleNamespace(status="pass")
    evidence = s.collect_model_evidence(
        fit=SimpleNamespace(family="binary"),
        design=component,
        posterior=object(),
        ppc={"status": "review"},
    )
    assert evidence.family == "binary"
    assert len(evidence.component_table) == 9
    assert evidence.component_table["available"].sum() == 3

    manifest_evidence = s.collect_model_evidence(manifest=SimpleNamespace(family="duration"))
    assert manifest_evidence.family == "duration"

    monkeypatch.setattr(unified, "diagnose_model_fit", lambda fit: component)
    monkeypatch.setattr(unified, "summarise_model_posterior", lambda fit: component)
    monkeypatch.setattr(unified, "estimate_model_estimands", lambda fit: component)
    computed = s.collect_model_evidence(
        fit=SimpleNamespace(family="binary"),
        compute=("diagnostics", "posterior", "estimands", "diagnostics"),
    )
    assert computed.component_table.loc[
        computed.component_table["component"].isin(["diagnostics", "posterior", "estimands"]),
        "available",
    ].all()

    with pytest.raises(GP3BayesError):
        s.collect_model_evidence(compute=("bad",))
    with pytest.raises(GP3BayesError):
        s.collect_model_evidence(compute=("diagnostics",))

    path = tmp_path / "evidence.md"
    result = s.create_model_evidence_report(evidence, path)
    assert path.is_file()
    assert "evidence inventory" in path.read_text(encoding="utf-8").lower()
    assert result == str(path.resolve())

    with pytest.raises(GP3BayesError):
        s.create_model_evidence_report(object(), tmp_path / "x.md")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        s.create_model_evidence_report(evidence, path)
    assert s.create_model_evidence_report(evidence, path, overwrite=True)
    with pytest.raises(GP3BayesError):
        s.create_model_evidence_report(evidence, tmp_path / "absent" / "x.md")


def test_publication_table_adapters_and_powerscale_sbc_forms():
    parameter = pd.DataFrame(
        {"variable": ["a"], "standardized_bias": [0.1], "coverage": [0.9], "rmse": [0.2]}
    )
    estimates = pd.DataFrame(
        {
            "variable": ["a"],
            "truth": [0.0],
            "median": [0.1],
            "lower": [-0.1],
            "upper": [0.3],
            "repetition": [1],
        }
    )
    fit_status = pd.DataFrame({"diagnostic_status": ["pass"], "completed": [True]})
    comparison = pd.DataFrame(
        {
            "scenario": ["wide"],
            "scale_multiplier": [2.0],
            "variable": ["a"],
            "standardized_shift": [0.1],
        }
    )
    scenarios = pd.DataFrame({"scenario": ["wide"], "maximum_standardized_shift": [0.1]})
    estimand = pd.DataFrame(
        {
            "alternative": ["alt"],
            "reference_median": [0.0],
            "alternative_median": [0.1],
            "alternative_lower": [-0.1],
            "alternative_upper": [0.3],
        }
    )
    deletion = pd.DataFrame({"omitted_unit": ["p1"], "median_shift": [0.05]})

    obj = SimpleNamespace(
        parameter_summary=parameter,
        estimates=estimates,
        fit_status=fit_status,
        comparison=comparison,
        scenario_status=scenarios,
        table=estimand,
        summary=deletion,
    )
    assert s.recovery_parameter_table(obj).equals(parameter)
    assert s.recovery_estimate_table(obj).equals(estimates)
    assert s.recovery_fit_status_table(obj).equals(fit_status)
    assert s.prior_sensitivity_table(obj).equals(comparison)
    assert s.prior_sensitivity_scenario_table(obj).equals(scenarios)
    assert s.estimand_sensitivity_table(obj).equals(estimand)
    assert s.group_deletion_sensitivity_table(obj).equals(deletion)
    assert s.random_slope_sensitivity_table({"comparison": obj}).equals(estimand)

    raw = pd.DataFrame({"component": ["prior"], "alpha": [0.9], "distance": [0.1]})
    assert s.powerscale_sensitivity_table(raw).equals(raw)
    assert s.powerscale_sensitivity_table({"raw": raw}).equals(raw)
    assert s.powerscale_sensitivity_table([[1, 2]]).shape == (1, 2)

    sbc_stats = pd.DataFrame(
        {
            "parameter": ["a", "a", "b"],
            "rank": [1, 5, 8],
            "draws": [10, 10, 10],
            "coverage": [0.9, 0.8, 0.9],
            "truth": [0.0, 0.0, 1.0],
            "median": [0.1, -0.1, 0.9],
        }
    )
    sbc = {
        "raw": {"stats": sbc_stats},
        "plan": {"family": "binary", "backend": "pymc", "n_sims": 3},
        "status": "completed",
        "diagnostics_inspected": True,
    }
    assert s.sbc_stats_table(sbc).equals(sbc_stats)
    overview = s.sbc_overview_table(sbc)
    assert overview.loc[0, "variables_recorded"] == 2

    with pytest.raises(GP3BayesError):
        s.random_slope_sensitivity_table({})
    with pytest.raises(GP3BayesError):
        s.sbc_stats_table({})
    with pytest.raises(GP3BayesError):
        s._sbc_plot_data(sbc, variables=("missing",))


def test_sensitivity_publication_plot_matrix_and_errors():
    parameter = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "standardized_bias": [0.1, -0.2],
            "coverage": [0.9, 0.8],
            "rmse": [0.2, 0.3],
        }
    )
    estimates = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "truth": [0.0, 1.0],
            "median": [0.1, 0.9],
            "lower": [-0.1, 0.7],
            "upper": [0.3, 1.1],
            "repetition": [1, 1],
        }
    )
    fit_status = pd.DataFrame(
        {
            "diagnostic_status": ["pass", "review"],
            "completed": [True, False],
        }
    )
    prior = pd.DataFrame(
        {
            "scenario": ["tight", "wide"],
            "scale_multiplier": [0.5, 2.0],
            "variable": ["a", "a"],
            "standardized_shift": [0.1, 0.2],
        }
    )
    scenarios = pd.DataFrame(
        {
            "scenario": ["tight", "wide"],
            "maximum_standardized_shift": [0.1, 0.2],
        }
    )
    estimand = pd.DataFrame(
        {
            "alternative": ["alt1", "alt2"],
            "reference_median": [0.0, 0.0],
            "alternative_median": [0.1, -0.1],
            "alternative_lower": [-0.1, -0.3],
            "alternative_upper": [0.3, 0.1],
        }
    )
    deletion = pd.DataFrame({"omitted_unit": ["p1", "p2"], "median_shift": [0.05, -0.03]})
    power_wide = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "prior": [0.1, 0.2],
            "likelihood": [0.05, 0.1],
        }
    )
    power_long = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "component": ["prior", "likelihood"],
            "alpha": [0.9, 1.1],
            "distance": [0.1, 0.2],
        }
    )
    sbc = {
        "raw": {
            "stats": pd.DataFrame(
                {
                    "parameter": ["a", "a", "b", "b"],
                    "rank": [1, 5, 2, 8],
                    "draws": [10, 10, 10, 10],
                    "coverage": [0.9, 0.8, 0.9, 0.85],
                    "truth": [0.0, 0.0, 1.0, 1.0],
                    "median": [0.1, -0.1, 0.9, 1.1],
                }
            )
        }
    }

    makers = (
        lambda: s.plot_recovery_bias(parameter),
        lambda: s.plot_recovery_coverage(parameter),
        lambda: s.plot_recovery_rmse(parameter),
        lambda: s.plot_recovery_estimates(estimates),
        lambda: s.plot_recovery_fit_status(fit_status),
        lambda: s.plot_prior_sensitivity(prior),
        lambda: s.plot_prior_sensitivity_scenarios(scenarios),
        lambda: s.plot_estimand_sensitivity_gg(estimand),
        lambda: s.plot_group_deletion_sensitivity(deletion),
        lambda: s.plot_random_slope_sensitivity(estimand),
        lambda: s.plot_powerscale_sensitivity_gg(power_wide),
        lambda: s.plot_powerscale_sensitivity_gg(power_long),
        lambda: s.plot_sbc_rank_gg(sbc),
        lambda: s.plot_sbc_ecdf_gg(sbc),
        lambda: s.plot_sbc_coverage_gg(sbc),
        lambda: s.plot_sbc_simulated_vs_estimated_gg(sbc),
    )
    for make in makers:
        fig = make()
        assert fig.axes
        plt.close(fig)

    # Rank-only SBC branches (coverage proxy and fallback scatter).
    rank_only = {
        "raw": {
            "stats": pd.DataFrame(
                {"variable": ["a", "b", "c"], "rank": [1, 4, 8], "max_rank": [10, 10, 10]}
            )
        }
    }
    for make in (
        lambda: s.plot_sbc_coverage_gg(rank_only),
        lambda: s.plot_sbc_simulated_vs_estimated_gg(rank_only),
    ):
        fig = make()
        assert fig.axes
        plt.close(fig)

    with pytest.raises(GP3BayesError):
        s.plot_recovery_bias(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_recovery_coverage(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_recovery_rmse(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_recovery_estimates(estimates, variables=("missing",))
    with pytest.raises(GP3BayesError):
        s.plot_recovery_fit_status(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_prior_sensitivity(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_prior_sensitivity_scenarios(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_estimand_sensitivity_gg(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_group_deletion_sensitivity(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_powerscale_sensitivity_gg(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        s.plot_sbc_rank_gg({"raw": {"stats": pd.DataFrame({"x": [1]})}})
    with pytest.raises(GP3BayesError):
        s.plot_sbc_simulated_vs_estimated_gg({"raw": {"stats": pd.DataFrame({"x": [1]})}})
