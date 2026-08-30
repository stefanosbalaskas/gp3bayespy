from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.reporting as reporting
import gp3bayespy.reproducibility as repro
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@dataclass
class DemoDataclass:
    x: int
    y: float


class DemoObject:
    def __init__(self):
        self.value = 3
        self.backend_private = "excluded"


class SlotOnly:
    __slots__ = ()


def _fake_spec(family: str = "binary"):
    data = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "selected": [0, 1, 1, 0],
            "x": [0.1, 0.2, 0.3, 0.4],
        }
    )
    contract = {"family": family, "outcome": "selected"}
    prepared = SimpleNamespace(data=data, transformations={"x": {"scaled": True}})
    return SimpleNamespace(
        family=family,
        model_family="demo",
        contract=contract,
        prepared=prepared,
        formula_text="selected ~ x",
    )


def test_reproducibility_stable_hash_and_fingerprint_branches(tmp_path):
    values = (
        None,
        "x",
        1,
        True,
        1.25,
        float("inf"),
        np.float64(2.5),
        tmp_path / "a.txt",
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.Series([1, 2], name="s"),
        np.array([[1, 2], [3, 4]]),
        DemoDataclass(1, 2.0),
        {"b": 2, "a": 1},
        (1, 2),
        [1, 2],
        {1, 2},
        DemoObject(),
        SlotOnly(),
    )
    stabilized = [repro._stable(value) for value in values]
    assert len(stabilized) == len(values)
    assert repro._hash({"a": 1}) == repro._hash({"a": 1})

    absent = repro._data_fingerprint(None)
    present = repro._data_fingerprint(pd.DataFrame({"a": [1, 2]}))
    assert not absent["available"]
    assert present["available"]
    with pytest.raises(GP3BayesError, match="DataFrame"):
        repro._data_fingerprint([1, 2])  # type: ignore[arg-type]

    assert not repro._signature(None)["available"]
    assert repro._signature({"a": 1})["available"]
    assert "numpy" in repro._versions()


def test_manifest_lifecycle_comparison_reporting_and_validation(tmp_path):
    spec = _fake_spec()
    manifest = repro.create_analysis_manifest(
        specification=spec,
        estimands=("contrast",),
        seed=7,
        label="analysis-a",
        notes=("registered",),
    )
    assert repro.validate_analysis_manifest(manifest).status == "pass"
    assert len(repro.analysis_manifest_table(manifest)) == 5

    frozen_path = tmp_path / "manifest.pkl"
    frozen = repro.freeze_analysis_manifest(manifest, frozen_path)
    assert frozen.frozen and frozen.manifest_hash
    loaded = repro.read_analysis_manifest(frozen_path)
    assert loaded.manifest_hash == frozen.manifest_hash

    report_path = tmp_path / "report.md"
    assert Path(repro.write_reproducibility_report(frozen, report_path)).is_file()

    same = repro.compare_analysis_manifests(frozen, loaded)
    assert same.identical

    changed = repro.create_analysis_manifest(
        specification=spec,
        estimands=("contrast", "auc"),
        seed=8,
        label="analysis-b",
    )
    changed_cmp = repro.compare_analysis_manifests(frozen, changed)
    assert not changed_cmp.identical
    assert changed_cmp.changed_components

    with pytest.raises(GP3BayesError, match="already exists"):
        repro.freeze_analysis_manifest(frozen, frozen_path)
    with pytest.raises(GP3BayesError, match="parent directory"):
        repro.freeze_analysis_manifest(
            repro.create_analysis_manifest(specification=spec),
            tmp_path / "missing" / "manifest.pkl",
        )
    with pytest.raises(GP3BayesError, match="does not exist"):
        repro.read_analysis_manifest(tmp_path / "absent.pkl")
    with pytest.raises(GP3BayesError, match="already exists"):
        repro.write_reproducibility_report(frozen, report_path)
    with pytest.raises(GP3BayesError, match="parent directory"):
        repro.write_reproducibility_report(frozen, tmp_path / "missing2" / "report.md")

    assert repro.validate_analysis_manifest(object()).status == "fail"  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="validation failed"):
        repro.validate_analysis_manifest(object(), strict=True)  # type: ignore[arg-type]


def test_manifest_creation_validation_errors_and_fit_inference():
    spec = _fake_spec()
    fit = SimpleNamespace(
        family="binary",
        specification=spec,
        sampling={"seed": 17, "chains": 2},
    )
    inferred = repro.create_analysis_manifest(fit=fit)
    assert inferred.family == "binary"
    assert inferred.seed == 17
    assert inferred.sampling["chains"] == 2

    with pytest.raises(GP3BayesError, match="approved gp3bayes fit"):
        repro.create_analysis_manifest(fit=SimpleNamespace(family="other"))
    with pytest.raises(GP3BayesError, match="approved gp3bayes model"):
        repro.create_analysis_manifest(specification=SimpleNamespace(family="other"))
    for seed in (-1, True, 1.2):
        with pytest.raises(GP3BayesError, match="seed"):
            repro.create_analysis_manifest(specification=spec, seed=seed)
    with pytest.raises(GP3BayesError, match="label"):
        repro.create_analysis_manifest(specification=spec, label="")
    with pytest.raises(GP3BayesError, match="notes"):
        repro.create_analysis_manifest(specification=spec, notes=("",))


def test_capture_analysis_bundle_publication_tables_figures_and_report(monkeypatch, tmp_path):
    import gp3bayespy.advanced_optional_workflows as aow
    import gp3bayespy.postfit_exploration as postfit
    import gp3bayespy.predictive as predictive

    fit = SimpleNamespace(
        family="binary",
        specification=_fake_spec(),
        sampling={"max_treedepth": 12},
    )

    posterior = pd.DataFrame(
        {"variable": ["b_x"], "lower": [-0.2], "median": [0.1], "upper": [0.4]}
    )
    monkeypatch.setattr(postfit, "posterior_interval_table", lambda *a, **k: posterior)
    monkeypatch.setattr(
        postfit,
        "summarise_mcmc_quality",
        lambda fit: SimpleNamespace(issues=pd.DataFrame({"status": ["pass"]}), status="pass"),
    )
    monkeypatch.setattr(
        postfit,
        "group_effect_table",
        lambda fit: pd.DataFrame({"level": ["p1"], "median": [0.1]}),
    )
    monkeypatch.setattr(
        postfit,
        "variance_component_table",
        lambda fit: pd.DataFrame({"component": ["sd"], "median": [0.2]}),
    )

    monkeypatch.setattr(
        predictive,
        "audit_prediction_support",
        lambda fit, data: SimpleNamespace(table=pd.DataFrame({"status": ["pass"]}), status="pass"),
    )

    expected = SimpleNamespace(
        observed=np.array([0, 1]),
        summary=pd.DataFrame(
            {
                "predicted_mean": [0.2, 0.8],
                "lower": [0.1, 0.7],
                "upper": [0.3, 0.9],
            }
        ),
    )
    predictive_draw = SimpleNamespace(
        observed=np.array([0, 1]),
        summary=expected.summary.copy(),
    )
    monkeypatch.setattr(
        predictive,
        "predict_model",
        lambda *a, **k: predictive_draw if k.get("type") == "predictive" else expected,
    )
    monkeypatch.setattr(
        predictive,
        "binary_prediction_scores",
        lambda x: pd.DataFrame({"metric": ["brier"], "value": [0.1]}),
    )
    monkeypatch.setattr(
        predictive,
        "binary_calibration_table",
        lambda x: pd.DataFrame(
            {
                "mean_predicted_probability": [0.2, 0.8],
                "observed_rate": [0.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        predictive,
        "predictive_coverage_table",
        lambda x: pd.DataFrame({"nominal_coverage": [0.8], "empirical_coverage": [1.0]}),
    )
    monkeypatch.setattr(
        aow,
        "compute_psis_loo",
        lambda fit: SimpleNamespace(
            __class__=type("PSISLOOResult", (), {}),
            pointwise=pd.DataFrame({"elpd_loo": [-1.0]}),
        ),
    )

    bundle = repro.create_analysis_bundle(
        fit, ndraws=20, include_group_effects=True, include_loo=True
    )
    assert bundle.family == "binary"
    assert len(repro.analysis_bundle_table(bundle)) >= 7
    tables = repro.create_publication_table_set(bundle)
    assert "posterior" in tables
    assert "scores" in tables

    monkeypatch.setattr(reporting, "plot_mcmc_quality", lambda x: plt.subplots()[0])
    monkeypatch.setattr(reporting, "plot_prediction_support", lambda x: plt.subplots()[0])
    monkeypatch.setattr(reporting, "plot_prediction_intervals", lambda x: plt.subplots()[0])
    monkeypatch.setattr(reporting, "plot_group_effects", lambda x: plt.subplots()[0])
    monkeypatch.setattr(reporting, "plot_variance_components", lambda x: plt.subplots()[0])
    monkeypatch.setattr(reporting, "plot_loo_influence", lambda x: plt.subplots()[0])
    monkeypatch.setattr(
        reporting,
        "create_figure_set",
        lambda figures=None, title="": reporting.FigureSet(
            title, figures or {}, tuple((figures or {}).keys())
        ),
    )

    figures = repro.create_analysis_figure_set(bundle)
    assert figures.figures

    report = repro.write_analysis_bundle_report(bundle, tmp_path / "bundle.md")
    assert Path(report).is_file()

    good = repro._capture(lambda x: x + 1, 1)
    bad = repro._capture(lambda: 1 / 0)
    assert good.ok and not bad.ok

    with pytest.raises(GP3BayesError):
        repro.create_analysis_bundle(SimpleNamespace(family="other"))
    with pytest.raises(GP3BayesError):
        repro.analysis_bundle_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        repro.create_publication_table_set(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        repro.create_analysis_figure_set(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        repro.write_analysis_bundle_report(object(), tmp_path / "x.md")  # type: ignore[arg-type]


def _figure():
    return plt.subplots()[0]


def test_reporting_theme_figure_set_registry_and_writers(tmp_path):
    assert reporting.theme_gp3bayes(12)["base_size"] == 12
    with pytest.raises(GP3BayesError, match="base_size"):
        reporting.theme_gp3bayes(0)

    fig1, fig2 = _figure(), _figure()
    figures = reporting.create_figure_set({"a": fig1, "b": fig2}, title="set")
    saved = reporting.save_figure_set(figures, tmp_path / "figures", dpi=60)
    assert len(saved) == 2
    with pytest.raises(GP3BayesError):
        reporting.create_figure_set({})
    with pytest.raises(GP3BayesError):
        reporting.create_figure_set({"": fig1})
    with pytest.raises(GP3BayesError):
        reporting.create_figure_set({"bad": object()})
    with pytest.raises(GP3BayesError):
        reporting.save_figure_set(object(), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="already exists"):
        reporting.save_figure_set(figures, tmp_path / "figures", dpi=60)

    registry = reporting.create_publication_registry("paper")
    registry = reporting.register_publication_table(
        registry, "table1", pd.DataFrame({"a": [1]}), caption="Table"
    )
    registry = reporting.register_publication_figure(
        registry, "figure1", _figure(), caption="Figure"
    )
    assert len(reporting.publication_registry_table(registry)) == 2
    assert reporting.validate_publication_registry(registry).valid

    registry_path = tmp_path / "registry.md"
    assert Path(reporting.write_publication_registry(registry, registry_path)).is_file()
    fig_files = reporting.save_publication_registry_figures(
        registry, tmp_path / "registry_figures", dpi=60
    )
    assert len(fig_files) == 1

    empty_registry = reporting.create_publication_registry()
    assert reporting.save_publication_registry_figures(empty_registry, tmp_path / "none").empty

    with pytest.raises(GP3BayesError):
        reporting.create_publication_registry("")
    with pytest.raises(GP3BayesError):
        reporting.register_publication_table(registry, "", pd.DataFrame())
    with pytest.raises(GP3BayesError):
        reporting.register_publication_table(registry, "bad", object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="already exists"):
        reporting.register_publication_table(registry, "table1", pd.DataFrame())
    with pytest.raises(GP3BayesError):
        reporting.register_publication_figure(registry, "bad", object())
    with pytest.raises(GP3BayesError, match="already exists"):
        reporting.register_publication_figure(registry, "figure1", _figure())
    with pytest.raises(GP3BayesError):
        reporting.publication_registry_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="already exists"):
        reporting.write_publication_registry(registry, registry_path)


def test_reporting_inventory_dashboard_model_card_and_reports(monkeypatch, tmp_path):
    inventory = reporting.create_complete_evidence_inventory(
        {"fit": SimpleNamespace(status="pass"), "loo": None}, label="inventory"
    )
    assert len(reporting.evidence_inventory_table(inventory)) == 2
    with pytest.raises(GP3BayesError):
        reporting.create_complete_evidence_inventory({})
    with pytest.raises(GP3BayesError):
        reporting.evidence_inventory_table(object())  # type: ignore[arg-type]

    dashboard = reporting.create_diagnostic_dashboard(
        fit=SimpleNamespace(status="pass"), recovery=SimpleNamespace(status="review")
    )
    assert len(reporting.diagnostic_dashboard_table(dashboard)) == 8
    assert reporting.plot_diagnostic_dashboard(dashboard).axes
    dashboard_path = tmp_path / "dashboard.md"
    assert Path(reporting.write_diagnostic_dashboard_report(dashboard, dashboard_path)).is_file()

    with pytest.raises(GP3BayesError):
        reporting.create_diagnostic_dashboard()
    with pytest.raises(GP3BayesError):
        reporting.diagnostic_dashboard_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        reporting.write_diagnostic_dashboard_report(
            object(),
            tmp_path / "bad.md",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError, match="already exists"):
        reporting.write_diagnostic_dashboard_report(dashboard, dashboard_path)

    import gp3bayespy.unified_workflow_api as unified

    monkeypatch.setattr(unified, "diagnose_model_fit", lambda fit: SimpleNamespace(status="pass"))
    monkeypatch.setattr(
        unified, "model_workflow_status", lambda fit: SimpleNamespace(status="complete")
    )
    fit = SimpleNamespace(
        family="binary",
        specification=SimpleNamespace(formula_text="selected ~ x", model_family="Bernoulli"),
        sampling_backend="pymc",
        sampling={"chains": 2},
        package_versions={"gp3bayespy": "0.5.0"},
    )
    card = reporting.create_model_card(
        fit, analysis_bundle=SimpleNamespace(status="available"), manifest=object()
    )
    assert len(reporting.model_card_table(card)) == 5
    checklist = reporting.create_reporting_checklist(card)
    assert len(checklist) == 9
    assert reporting.plot_reporting_checklist(checklist).axes
    card_path = tmp_path / "card.md"
    assert Path(reporting.write_model_card(card, card_path)).is_file()

    with pytest.raises(GP3BayesError):
        reporting.create_model_card(SimpleNamespace(family="other"))
    with pytest.raises(GP3BayesError):
        reporting.model_card_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        reporting.write_model_card(object(), tmp_path / "bad-card.md")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="already exists"):
        reporting.write_model_card(card, card_path)


def test_reporting_plot_adapter_matrix():
    posterior = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "lower": [-0.4, 0.1],
            "median": [0.0, 0.5],
            "upper": [0.4, 0.9],
        }
    )
    draws = pd.DataFrame(
        {
            "a": np.linspace(-1, 1, 120),
            "b": np.linspace(0, 2, 120),
            "c": np.sin(np.linspace(0, 4, 120)),
        }
    )
    mcmc = pd.DataFrame({"status": ["pass", "review", "pass"]})
    pred = SimpleNamespace(
        summary=pd.DataFrame(
            {
                "predicted_mean": [0.2, 0.5, 0.8],
                "lower": [0.1, 0.4, 0.7],
                "upper": [0.3, 0.6, 0.9],
            }
        )
    )
    binary_cal = pd.DataFrame(
        {
            "mean_predicted_probability": [0.2, 0.5, 0.8],
            "observed_rate": [0.1, 0.6, 0.9],
        }
    )
    threshold = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "accuracy": [0.7, 0.8, 0.75],
            "sensitivity": [0.9, 0.8, 0.6],
            "specificity": [0.5, 0.8, 0.9],
            "balanced_accuracy": [0.7, 0.8, 0.75],
        }
    )
    qcal = SimpleNamespace(
        table=pd.DataFrame(
            {
                "probability": [0.1, 0.5, 0.9],
                "empirical_probability": [0.12, 0.48, 0.88],
            }
        )
    )
    pit = SimpleNamespace(table=pd.DataFrame({"pit": np.linspace(0.05, 0.95, 20)}))
    exceed = pd.DataFrame({"probability": [0.1, 0.4, 0.9]})
    coverage = pd.DataFrame(
        {
            "nominal_coverage": [0.5, 0.8, 0.95],
            "empirical_coverage": [0.52, 0.79, 0.94],
        }
    )
    residual = pd.DataFrame({"residual": [-0.2, 0.0, 0.3]})
    support = SimpleNamespace(table=pd.DataFrame({"status": ["pass", "review", "pass"]}))
    uncertainty = SimpleNamespace(
        table=pd.DataFrame(
            {
                "epistemic_variance": [0.1, 0.2],
                "residual_variance": [0.3, 0.4],
                "total_variance": [0.4, 0.6],
            }
        )
    )
    grouped = SimpleNamespace(
        table=pd.DataFrame(
            {
                "group": ["a", "b"],
                "observed": [1.0, 2.0],
                "predicted_mean": [1.1, 1.9],
            }
        )
    )
    group_effects = pd.DataFrame(
        {"group": ["participant", "participant"], "level": ["p1", "p2"], "median": [0.1, -0.2]}
    )
    variances = pd.DataFrame({"component": ["participant", "residual"], "median": [0.2, 0.5]})
    loo = pd.DataFrame({"pareto_k": [0.2, 0.8, 0.4]})
    comparison = pd.DataFrame({"model": ["m1", "m2"], "elpd_loo": [-10, -12]})
    weights = pd.DataFrame({"model": ["m1", "m2"], "weight": [0.7, 0.3]})

    figures = (
        reporting.plot_posterior_intervals(posterior),
        reporting.plot_posterior_areas(posterior),
        reporting.plot_posterior_density(draws),
        reporting.plot_posterior_correlations(draws),
        reporting.plot_mcmc_quality(mcmc),
        reporting.plot_prediction_intervals(pred),
        reporting.plot_binary_calibration(binary_cal),
        reporting.plot_binary_threshold_metrics(threshold),
        reporting.plot_duration_quantile_calibration(qcal),
        reporting.plot_duration_pit(pit),
        reporting.plot_exceedance_probability(exceed),
        reporting.plot_predictive_coverage(coverage),
        reporting.plot_predictive_residuals(residual),
        reporting.plot_prediction_support(support),
        reporting.plot_uncertainty_decomposition(uncertainty),
        reporting.plot_grouped_prediction_check(grouped),
        reporting.plot_group_effects(group_effects, groups=("participant",)),
        reporting.plot_variance_components(variances),
        reporting.plot_loo_influence(loo),
        reporting.plot_model_comparison(comparison),
        reporting.plot_model_weights(weights),
    )
    assert all(fig is not None and fig.axes for fig in figures)
    for fig in figures:
        plt.close(fig)

    one_d = reporting._posterior_frame(np.arange(10.0))
    two_d = reporting._posterior_frame(np.arange(20.0).reshape(10, 2))
    assert len(one_d) == 1 and len(two_d) == 2
    assert reporting._status(None) == "not_available"
    assert reporting._status({}) == "available"
    assert reporting._status({"status": "review"}) == "review"
    with pytest.raises(GP3BayesError):
        reporting._df(object(), "table")
