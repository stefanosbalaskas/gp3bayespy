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

s = importlib.import_module("gp3bayespy.sensitivity")
rp = importlib.import_module("gp3bayespy.reproducibility")
rep = importlib.import_module("gp3bayespy.reporting")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _binary_spec(seed=5201):
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


def test_sensitivity_duration_unit_reference_success_and_error(monkeypatch):
    u = importlib.import_module("gp3bayespy.unified_workflow_api")
    closure = importlib.import_module("gp3bayespy.specification_closure")

    reference = SimpleNamespace(status="pass")
    converted = SimpleNamespace(status="pass")
    result = SimpleNamespace(status="pass", interpretation="ok")

    monkeypatch.setattr(u, "estimate_model_estimands", lambda fit: reference)
    monkeypatch.setattr(closure, "audit_duration_unit_invariance", lambda **kwargs: result)

    plan = s.create_sensitivity_suite_plan(
        duration_unit={"estimand": converted, "multiplier": 0.001}
    )
    suite = s.run_sensitivity_suite(SimpleNamespace(family="duration"), plan)
    assert suite.status == "completed"
    assert suite.reference_estimand is reference
    assert suite.results["duration_unit"] is result

    monkeypatch.setattr(
        u,
        "estimate_model_estimands",
        lambda fit: (_ for _ in ()).throw(RuntimeError("synthetic estimand failure")),
    )
    failed = s.run_sensitivity_suite(
        SimpleNamespace(family="duration"),
        plan,
        stop_on_error=False,
    )
    assert isinstance(failed.reference_estimand, s.SuiteError)
    assert failed.status == "not_run"


def test_sensitivity_raw_powerscale_object_sbc_object_and_rank_fallback():
    raw_holder = SimpleNamespace(raw=[{"variable": "b", "prior": 0.1, "likelihood": 0.2}])
    assert not s.powerscale_sensitivity_table(raw_holder).empty

    with pytest.raises(GP3BayesError):
        s.powerscale_sensitivity_table(object())

    sbc_object = SimpleNamespace(
        raw={"stats": [{"rank": 1}, {"rank": 2}]},
        plan=SimpleNamespace(family="binary", backend="analytic", n_sims=2),
        status="completed",
        diagnostics_inspected=False,
    )
    overview = s.sbc_overview_table(sbc_object)
    assert overview.iloc[0]["family"] == "binary"
    assert np.isnan(overview.iloc[0]["variables_recorded"])

    mapping = {"raw": {"stats": [{"rank": 1}, {"rank": 2}, {"rank": 3}]}}
    assert s.plot_sbc_rank_gg(mapping).axes
    assert s.plot_sbc_ecdf_gg(mapping).axes
    assert s.plot_sbc_coverage_gg(mapping).axes


def test_repro_manifest_fit_derivation_and_psis_publication_table():
    spec = _binary_spec()
    fit = SimpleNamespace(
        family="binary",
        specification=spec,
        sampling={"seed": 123},
    )
    manifest = rp.create_analysis_manifest(fit=fit)
    assert manifest.family == "binary"
    assert manifest.seed == 123
    assert manifest.data["available"] is True

    PSISLOOResult = type("PSISLOOResult", (), {})
    loo_value = PSISLOOResult()
    loo_value.estimates = pd.DataFrame(
        {"estimate": [1.0], "se": [0.1]},
        index=["elpd_loo"],
    )
    captured = rp.CapturedComponent(True, loo_value, None)
    bundle = rp.AnalysisBundle(
        "0.3",
        "binary",
        fit,
        {"loo": captured},
        pd.DataFrame([{"component": "loo", "available": True, "error": ""}]),
        True,
    )
    tables = rp.create_publication_table_set(bundle)
    assert "loo" in tables
    assert "quantity" in tables["loo"].columns


def test_reporting_empty_title_mapping_fallback_and_predictive_adapters(monkeypatch):
    fig, ax = rep._figure("")
    assert ax.get_title() == ""

    frame = pd.DataFrame({"mean": [1.0, 2.0]})
    assert rep._df({"summary": frame}, "summary").equals(frame)

    pe = importlib.import_module("gp3bayespy.postfit_exploration")
    fallback = pd.DataFrame(
        {
            "variable": ["b"],
            "lower": [0.0],
            "median": [1.0],
            "upper": [2.0],
        }
    )
    monkeypatch.setattr(pe, "posterior_interval_table", lambda *a, **k: fallback)
    assert rep._posterior_frame(SimpleNamespace()).equals(fallback)

    predictive = importlib.import_module("gp3bayespy.predictive")
    calibration = pd.DataFrame(
        {
            "mean_predicted_probability": [0.2, 0.8],
            "observed_rate": [0.1, 0.9],
        }
    )
    monkeypatch.setattr(predictive, "binary_calibration_table", lambda *a, **k: calibration)
    assert rep.plot_binary_calibration(SimpleNamespace()).axes

    thresholds = pd.DataFrame(
        {
            "threshold": [0.5],
            "accuracy": [0.8],
            "sensitivity": [0.7],
            "specificity": [0.9],
            "balanced_accuracy": [0.8],
        }
    )
    monkeypatch.setattr(predictive, "binary_threshold_metrics", lambda *a, **k: thresholds)
    assert rep.plot_binary_threshold_metrics(SimpleNamespace()).axes
