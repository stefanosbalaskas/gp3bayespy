from __future__ import annotations

import importlib
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")

rp = importlib.import_module("gp3bayespy.reproducibility")
rep = importlib.import_module("gp3bayespy.reporting")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _cc(ok=True, value=None, error=None):
    return rp.CapturedComponent(ok, value, error)


def test_repro_capture_publication_bundle_tables_and_report(tmp_path):
    assert rp._capture(lambda x: x + 1, 1).ok
    failed = rp._capture(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert not failed.ok and failed.error == "boom"

    status = pd.DataFrame(
        {
            "component": ["df", "summary", "issues", "table", "failed"],
            "available": [True, True, True, True, False],
            "error": ["", "", "", "", "x"],
        }
    )
    frame = pd.DataFrame({"x": [1]})
    bundle = rp.AnalysisBundle(
        "0.3",
        "binary",
        SimpleNamespace(),
        {
            "df": _cc(True, frame),
            "summary": _cc(True, SimpleNamespace(summary=frame)),
            "issues": _cc(True, SimpleNamespace(issues=frame)),
            "table": _cc(True, SimpleNamespace(table=frame)),
            "failed": _cc(False, None, "x"),
        },
        status,
        False,
    )
    tables = rp.create_publication_table_set(bundle)
    assert set(tables) == {"df", "summary", "issues", "table"}
    assert rp.analysis_bundle_table(bundle).equals(status)

    with pytest.raises(GP3BayesError):
        rp.analysis_bundle_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        rp.create_publication_table_set(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        rp.create_analysis_figure_set(object())  # type: ignore[arg-type]

    nofig = rp.AnalysisBundle(
        "0.3",
        "binary",
        SimpleNamespace(),
        {
            "mcmc": _cc(False),
            "prediction_support": _cc(False),
            "expected_prediction": _cc(False),
            "group_effects": _cc(False),
            "variance_components": _cc(False),
        },
        pd.DataFrame({"component": [], "available": []}),
        False,
    )
    with pytest.raises(GP3BayesError):
        rp.create_analysis_figure_set(nofig)

    out = tmp_path / "bundle.md"
    returned = rp.write_analysis_bundle_report(bundle, out)
    assert returned == str(out.resolve())
    assert "post-fit analysis bundle" in out.read_text(encoding="utf-8")
    with pytest.raises(GP3BayesError):
        rp.write_analysis_bundle_report(object(), out)  # type: ignore[arg-type]


def test_create_analysis_bundle_all_optional_branches(monkeypatch):
    pe = importlib.import_module("gp3bayespy.postfit_exploration")
    pr = importlib.import_module("gp3bayespy.predictive")
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")

    frame = pd.DataFrame({"x": [1.0, 2.0], "observed": [0, 1]})
    prediction = SimpleNamespace(
        observed=pd.Series([0, 1]),
        summary=pd.DataFrame({"predicted_mean": [0.2, 0.8]}),
    )

    monkeypatch.setattr(pe, "posterior_interval_table", lambda *a, **k: frame)
    monkeypatch.setattr(pe, "summarise_mcmc_quality", lambda *a, **k: SimpleNamespace(issues=frame))
    monkeypatch.setattr(pe, "group_effect_table", lambda *a, **k: frame)
    monkeypatch.setattr(pe, "variance_component_table", lambda *a, **k: frame)

    monkeypatch.setattr(
        pr, "audit_prediction_support", lambda *a, **k: SimpleNamespace(table=frame)
    )
    monkeypatch.setattr(pr, "predict_model", lambda *a, **k: prediction)
    monkeypatch.setattr(pr, "binary_prediction_scores", lambda *a, **k: frame)
    monkeypatch.setattr(pr, "duration_prediction_scores", lambda *a, **k: frame)
    monkeypatch.setattr(pr, "binary_calibration_table", lambda *a, **k: frame)
    monkeypatch.setattr(pr, "predictive_coverage_table", lambda *a, **k: frame)
    monkeypatch.setattr(pr, "duration_quantile_calibration", lambda *a, **k: frame)
    monkeypatch.setattr(aow, "compute_psis_loo", lambda *a, **k: SimpleNamespace(table=frame))

    with pytest.raises(GP3BayesError):
        rp.create_analysis_bundle(SimpleNamespace(family="pupil"))

    for family in ("binary", "duration"):
        fit = SimpleNamespace(
            family=family,
            specification=SimpleNamespace(
                prepared=SimpleNamespace(data=pd.DataFrame({"y": [0, 1]}))
            ),
        )
        bundle = rp.create_analysis_bundle(
            fit,
            ndraws=10,
            include_group_effects=True,
            include_loo=True,
        )
        names = set(bundle.components)
        assert {"scores", "coverage", "loo"}.issubset(names)
        if family == "binary":
            assert "calibration" in names
        else:
            assert "quantile_calibration" in names


def test_repro_figure_set_all_branches(monkeypatch):
    def fig(*args, **kwargs):
        return plt.figure()

    monkeypatch.setattr(rep, "plot_mcmc_quality", fig)
    monkeypatch.setattr(rep, "plot_prediction_support", fig)
    monkeypatch.setattr(rep, "plot_prediction_intervals", fig)
    monkeypatch.setattr(rep, "plot_group_effects", fig)
    monkeypatch.setattr(rep, "plot_variance_components", fig)
    monkeypatch.setattr(rep, "plot_loo_influence", fig)

    components = {
        name: _cc(True, SimpleNamespace())
        for name in (
            "mcmc",
            "prediction_support",
            "expected_prediction",
            "group_effects",
            "variance_components",
            "loo",
        )
    }
    status = pd.DataFrame(
        {
            "component": list(components),
            "available": [True] * len(components),
            "error": [""] * len(components),
        }
    )
    bundle = rp.AnalysisBundle("0.3", "binary", SimpleNamespace(), components, status, True)
    figures = rp.create_analysis_figure_set(bundle)
    assert len(figures.figures) == 6


def test_reporting_dashboard_figures_and_plot_fallbacks(monkeypatch, tmp_path):
    pb = importlib.import_module("gp3bayespy.prior_posterior_bridge")
    s = importlib.import_module("gp3bayespy.sensitivity")

    def fig(*args, **kwargs):
        return plt.figure()

    monkeypatch.setattr(rep, "plot_posterior_intervals", fig)
    monkeypatch.setattr(rep, "plot_reporting_checklist", fig)
    monkeypatch.setattr(rep, "plot_loo_influence", fig)
    monkeypatch.setattr(pb, "plot_prior_posterior_shift", fig)
    monkeypatch.setattr(pb, "plot_prior_posterior_contraction", fig)
    monkeypatch.setattr(s, "plot_recovery_bias", fig)
    monkeypatch.setattr(s, "plot_recovery_coverage", fig)

    dash = rep.create_diagnostic_dashboard(
        fit=SimpleNamespace(),
        model_card=SimpleNamespace(),
        prior_posterior=SimpleNamespace(),
        loo=SimpleNamespace(),
        recovery=SimpleNamespace(),
        label="all",
    )
    figs = rep.create_diagnostic_dashboard_figures(dash)
    assert len(figs.figures) == 7

    with pytest.raises(GP3BayesError):
        rep.create_diagnostic_dashboard_figures(object())  # type: ignore[arg-type]

    arr1 = rep._posterior_frame([1.0, 2.0, 3.0])
    assert len(arr1) == 1

    summary = pd.DataFrame({"mean": [1.0, 2.0]})
    assert rep.plot_prediction_intervals(summary).axes

    duration = pd.DataFrame({"probability": [0.25, 0.5], "empirical_probability": [0.2, 0.55]})
    assert rep.plot_duration_quantile_calibration(duration).axes

    report = tmp_path / "dash.md"
    assert rep.write_diagnostic_dashboard_report(dash, report) == str(report.resolve())
    rep.write_diagnostic_dashboard_report(dash, report, overwrite=True)
