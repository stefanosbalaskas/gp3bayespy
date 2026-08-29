from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy import design_support_diagnostics as ds
from gp3bayespy import evidence_graphics_gg as eg
from gp3bayespy import loo as loo_mod
from gp3bayespy import postfit_exploration as pe
from gp3bayespy import reporting as rep
from gp3bayespy import reproducibility as repro
from gp3bayespy.exceptions import GP3BayesError


class DummyLOO:
    def __init__(self) -> None:
        self.pointwise = pd.DataFrame(
            {
                "elpd_loo": [-1.1, -0.8, -1.4, -0.9, -1.7, -1.0],
                "mcse_elpd_loo": [0.08, 0.05, 0.10, 0.06, 0.12, 0.07],
            }
        )
        self.pareto_k = np.array([0.20, 0.45, 0.72, 0.35, 1.05, 0.63])
        self.influence_pareto_k = self.pareto_k.copy()


def _close(*figures):
    for fig in figures:
        if fig is not None:
            plt.close(fig)


def test_loo_tables_grouping_atlas_and_figures():
    x = DummyLOO()
    meta = pd.DataFrame({"participant": ["p1", "p1", "p2", "p2", "p3", "p3"]})
    table = loo_mod.loo_pointwise_table(x, data=meta)
    assert len(table) == 6
    assert table["flagged"].sum() == 2
    assert loo_mod.loo_influence_summary(table).loc[0, "flagged_k_ge_0_7"] == 2
    assert len(loo_mod.loo_flagged_data(table, threshold=0.7)) == 2
    grouped = loo_mod.loo_group_influence_table(table, "participant")
    atlas = loo_mod.create_loo_influence_atlas(x, data=meta)
    assert len(grouped) == 3
    assert len(loo_mod.loo_influence_atlas_table(atlas)) == 6
    figures = (
        loo_mod.plot_loo_pointwise_elpd(atlas),
        loo_mod.plot_loo_pareto_vs_elpd(atlas),
        loo_mod.plot_loo_influence_rank(atlas),
        loo_mod.plot_loo_group_influence(grouped),
        loo_mod.plot_loo_group_elpd(grouped),
    )
    assert all(fig.axes for fig in figures)
    _close(*figures)
    with pytest.raises(GP3BayesError, match="finite"):
        loo_mod.loo_flagged_data(table, threshold=float("nan"))
    with pytest.raises(GP3BayesError, match="group"):
        loo_mod.loo_group_influence_table(table, "missing")
    with pytest.raises(GP3BayesError, match="LOO influence atlas"):
        loo_mod.loo_influence_atlas_table(object())


def test_evidence_graphics_adapters_and_fallbacks():
    status = pd.DataFrame(
        {"component": ["posterior", "loo", "sensitivity"], "status": ["pass", "review", "pass"]}
    )
    parity = pd.DataFrame(
        {
            "variable": ["b_0", "b_1", "sigma"],
            "reference_mean": [0.1, 0.5, 1.0],
            "alternative_mean": [0.12, 0.48, 1.03],
        }
    )
    manifest = pd.DataFrame({"component": ["data", "contract", "seed"], "identical": [1, 1, 0]})
    missing = pd.DataFrame(
        {"variable": ["pupil", "gaze_x", "gaze_y"], "missing_fraction": [0.02, 0.08, 0.04]}
    )
    design = pd.DataFrame(
        {"check": ["rank", "conditioning", "repetition"], "status": ["pass", "review", "pass"]}
    )
    figures = (
        eg.plot_sensitivity_suite_gg({"table": status}),
        eg.plot_model_evidence_gg({"table": status}),
        eg.plot_backend_parity_gg({"table": parity}),
        eg.plot_backend_parity_gg({"table": status}),
        eg.plot_backend_environment_gg({"table": status}),
        eg.plot_manifest_comparison_gg({"table": manifest}),
        eg.plot_schema_comparison_gg({"table": status}),
        eg.plot_design_support_gg({"table": design}),
        eg.plot_missingness_gg({"table": missing}),
        eg.plot_missingness_gg({"table": pd.DataFrame({"note": ["recorded"]})}),
    )
    assert all(fig.axes for fig in figures)
    _close(*figures)
    with pytest.raises(GP3BayesError, match="requested table"):
        eg.model_evidence_table(object())


def test_reporting_registry_inventory_dashboard_and_save_paths(tmp_path: Path):
    assert rep.theme_gp3bayes()["base_size"] == 11
    with pytest.raises(GP3BayesError, match="strictly positive"):
        rep.theme_gp3bayes(0)

    fig = rep.plot_reporting_checklist(
        pd.DataFrame({"item": ["contract", "diagnostics"], "available": [1, 0]})
    )
    figure_set = rep.create_figure_set({"checklist": fig}, title="validation")
    saved = rep.save_figure_set(figure_set, tmp_path / "figs", dpi=80)
    assert Path(saved.loc[0, "file"]).exists()
    with pytest.raises(GP3BayesError, match="already exists"):
        rep.save_figure_set(figure_set, tmp_path / "figs", dpi=80)
    with pytest.raises(GP3BayesError, match="non-empty mapping"):
        rep.create_figure_set({})
    with pytest.raises(GP3BayesError, match="Matplotlib Figure"):
        rep.create_figure_set({"bad": object()})

    registry = rep.create_publication_registry("paper")
    registry = rep.register_publication_table(
        registry, "diagnostics", pd.DataFrame({"metric": ["rhat"], "value": [1.0]})
    )
    registry = rep.register_publication_figure(registry, "checklist", fig)
    assert len(rep.publication_registry_table(registry)) == 2
    assert rep.validate_publication_registry(registry).valid
    assert Path(rep.write_publication_registry(registry, tmp_path / "registry.md")).exists()
    saved_registry = rep.save_publication_registry_figures(
        registry, tmp_path / "registry-figures", dpi=80
    )
    assert len(saved_registry) == 1
    with pytest.raises(GP3BayesError, match="already exists"):
        rep.register_publication_table(registry, "diagnostics", pd.DataFrame())

    inventory = rep.create_complete_evidence_inventory(
        {"posterior": {"status": "pass"}, "loo": None}, label="inventory"
    )
    assert rep.evidence_inventory_table(inventory)["available"].tolist() == [True, False]
    with pytest.raises(GP3BayesError, match="non-empty mapping"):
        rep.create_complete_evidence_inventory({})

    dashboard = rep.create_diagnostic_dashboard(
        loo={"status": "review"}, sensitivity={"status": "pass"}, label="Evidence dashboard"
    )
    assert len(rep.diagnostic_dashboard_table(dashboard)) == 8
    dash = rep.plot_diagnostic_dashboard(dashboard)
    assert dash.axes
    _close(dash, fig)
    with pytest.raises(GP3BayesError, match="at least one"):
        rep.create_diagnostic_dashboard()


def test_reporting_plot_adapter_matrix():
    interval = pd.DataFrame(
        {
            "variable": ["b_0", "b_condition", "sigma"],
            "lower": [-0.3, 0.1, 0.7],
            "median": [0.0, 0.5, 1.0],
            "upper": [0.3, 0.9, 1.3],
            "mean": [0.01, 0.52, 1.01],
            "sd": [0.15, 0.20, 0.14],
        }
    )
    rng = np.random.default_rng(77)
    draws = pd.DataFrame(
        {
            "b_0": rng.normal(0.0, 0.25, 250),
            "b_condition": rng.normal(0.55, 0.20, 250),
            "sigma": rng.lognormal(-0.05, 0.12, 250),
        }
    )
    prediction = pd.DataFrame(
        {
            "predicted_mean": [0.15, 0.35, 0.72, 0.88],
            "lower": [0.05, 0.20, 0.58, 0.76],
            "upper": [0.28, 0.50, 0.84, 0.96],
        }
    )
    calibration = pd.DataFrame(
        {
            "mean_predicted_probability": [0.1, 0.3, 0.7, 0.9],
            "observed_rate": [0.08, 0.35, 0.68, 0.92],
        }
    )
    thresholds = pd.DataFrame(
        {
            "threshold": [0.2, 0.4, 0.6, 0.8],
            "accuracy": [0.65, 0.78, 0.82, 0.74],
            "sensitivity": [0.96, 0.88, 0.76, 0.52],
            "specificity": [0.36, 0.68, 0.87, 0.96],
            "balanced_accuracy": [0.66, 0.78, 0.82, 0.74],
        }
    )
    quantiles = pd.DataFrame(
        {
            "probability": [0.1, 0.25, 0.5, 0.75, 0.9],
            "empirical_probability": [0.12, 0.24, 0.51, 0.73, 0.88],
        }
    )
    figures = (
        rep.plot_posterior_intervals(interval),
        rep.plot_posterior_areas(interval),
        rep.plot_posterior_density(draws),
        rep.plot_posterior_correlations(draws),
        rep.plot_mcmc_quality(pd.DataFrame({"status": ["pass", "review", "pass"]})),
        rep.plot_prediction_intervals(prediction),
        rep.plot_binary_calibration(calibration),
        rep.plot_binary_threshold_metrics(thresholds),
        rep.plot_duration_quantile_calibration(quantiles),
        rep.plot_duration_pit(pd.DataFrame({"pit": [0.05, 0.15, 0.4, 0.55, 0.8, 0.92]})),
        rep.plot_exceedance_probability(pd.DataFrame({"probability": [0.1, 0.35, 0.7, 0.9]})),
        rep.plot_predictive_coverage(
            pd.DataFrame(
                {"nominal_coverage": [0.5, 0.8, 0.95], "empirical_coverage": [0.53, 0.79, 0.93]}
            )
        ),
        rep.plot_predictive_residuals(pd.DataFrame({"residual": [-0.4, 0.1, 0.25, -0.08]})),
        rep.plot_prediction_support(pd.DataFrame({"status": ["within", "review", "within"]})),
        rep.plot_uncertainty_decomposition(
            pd.DataFrame(
                {
                    "epistemic_variance": [0.02, 0.04, 0.03],
                    "residual_variance": [0.08, 0.07, 0.09],
                    "total_variance": [0.10, 0.11, 0.12],
                }
            )
        ),
        rep.plot_grouped_prediction_check(
            pd.DataFrame(
                {
                    "group": ["A", "B", "C"],
                    "observed": [0.2, 0.5, 0.8],
                    "predicted_mean": [0.23, 0.48, 0.77],
                }
            )
        ),
        rep.plot_group_effects(
            pd.DataFrame({"level": ["A", "B", "C"], "median": [-0.2, 0.1, 0.45]})
        ),
        rep.plot_variance_components(
            pd.DataFrame(
                {"component": ["participant", "item", "residual"], "variance": [0.4, 0.2, 0.8]}
            )
        ),
        rep.plot_model_comparison(
            pd.DataFrame({"model": ["M1", "M2", "M3"], "elpd_loo": [-105.2, -101.7, -103.1]})
        ),
        rep.plot_model_weights(
            pd.DataFrame({"model": ["M1", "M2", "M3"], "weight": [0.18, 0.62, 0.20]})
        ),
    )
    assert all(fig.axes for fig in figures)
    _close(*figures)


def test_postfit_tables_and_validation_branches():
    rng = np.random.default_rng(44)
    draws = pd.DataFrame(
        {
            "b_0": rng.normal(0, 1, 120),
            "b_1": rng.normal(0.5, 0.7, 120),
            "sigma": rng.lognormal(0, 0.1, 120),
        }
    )
    assert len(pe.posterior_interval_table(draws)) == 3
    assert "probability_in_rope" in pe.posterior_probability_table(draws, rope=(-0.1, 0.1))
    assert len(pe.posterior_correlation_table(draws, method="pearson")) == 3
    assert len(pe.posterior_correlation_table(draws, method="spearman")) == 3
    assert len(pe.posterior_interval_table(np.arange(20.0))) == 1
    assert len(pe.posterior_interval_table({"a": np.arange(10.0), "b": np.arange(10.0) + 1})) == 2
    assert len(pe.posterior_interval_table(draws, regex=r"^b_")) == 2

    diagnostics = pd.DataFrame(
        {
            "variable": ["b_0", "b_1"],
            "sd": [1.0, 0.5],
            "rhat": [1.0, 1.02],
            "ess_bulk": [800, 250],
            "ess_tail": [750, 200],
            "mcse_mean": [0.02, 0.08],
        }
    )
    assert pe.identify_mcmc_issues(diagnostics)["flagged"].tolist() == [False, True]

    with pytest.raises(GP3BayesError, match="Unknown posterior"):
        pe.posterior_interval_table(draws, variables=["missing"])
    with pytest.raises(GP3BayesError, match="regular expression"):
        pe.posterior_interval_table(draws, regex="[")
    with pytest.raises(GP3BayesError, match="wrong length"):
        pe.posterior_interval_table(draws, probs=(0.1, 0.9))
    with pytest.raises(GP3BayesError, match="rope"):
        pe.posterior_probability_table(draws, rope=(1, 0))
    with pytest.raises(GP3BayesError, match="pearson"):
        pe.posterior_correlation_table(draws, method="kendall")
    with pytest.raises(GP3BayesError, match="At least two"):
        pe.posterior_correlation_table(draws[["b_0"]])
    with pytest.raises(GP3BayesError, match="numeric 2-D"):
        pe.posterior_interval_table(np.ones((2, 2, 2)))


def test_reproducibility_roundtrip_and_errors(tmp_path: Path):
    data = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2"],
            "condition": ["A", "B", "A", "B"],
            "y": [0, 1, 1, 0],
        }
    )
    first = repro.create_analysis_manifest(
        data=data,
        estimands=["condition"],
        seed=41,
        label="analysis",
        notes=["deterministic synthetic data"],
    )
    assert repro.validate_analysis_manifest(first).status == "pass"
    assert len(repro.analysis_manifest_table(first)) == 5
    frozen = repro.freeze_analysis_manifest(first, tmp_path / "manifest.pkl")
    roundtrip = repro.read_analysis_manifest(tmp_path / "manifest.pkl")
    assert roundtrip.manifest_hash == frozen.manifest_hash
    second = repro.create_analysis_manifest(
        data=data.assign(y=[0, 1, 0, 0]),
        estimands=["condition"],
        seed=41,
        label="analysis",
        notes=["deterministic synthetic data"],
    )
    comparison = repro.compare_analysis_manifests(frozen, second)
    assert not comparison.identical
    assert "data_hash" in comparison.changed_components
    assert Path(
        repro.write_reproducibility_report(frozen, tmp_path / "reproducibility.md")
    ).exists()

    with pytest.raises(GP3BayesError, match="DataFrame"):
        repro.create_analysis_manifest(data=[1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="seed"):
        repro.create_analysis_manifest(data=data, seed=-1)
    with pytest.raises(GP3BayesError, match="label"):
        repro.create_analysis_manifest(data=data, label="")
    with pytest.raises(GP3BayesError, match="notes"):
        repro.create_analysis_manifest(data=data, notes=[""])
    with pytest.raises(GP3BayesError, match="already exists"):
        repro.freeze_analysis_manifest(frozen, tmp_path / "manifest.pkl")
    with pytest.raises(GP3BayesError, match="does not exist"):
        repro.read_analysis_manifest(tmp_path / "missing.pkl")
    with pytest.raises(GP3BayesError, match="already exists"):
        repro.write_reproducibility_report(frozen, tmp_path / "reproducibility.md")


def test_design_support_branches():
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        trial_col="trial_id",
        condition_col="condition",
    )
    data = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2"],
            "trial_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "condition": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "selected": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    assert ds.audit_missingness_structure(data, contract).status == "pass"
    changed = data.copy()
    changed.loc[0:1, "condition"] = np.nan
    assert ds.audit_missingness_structure(
        changed, contract, review_fraction=0.1, fail_fraction=0.5
    ).status in {"review", "fail"}
    assert ds.audit_fixed_effect_design(data, contract).n_rows == len(data)
    assert len(ds.audit_random_effects_support(data, contract).component_table) >= 1

    with pytest.raises(GP3BayesError, match="Missingness thresholds"):
        ds.audit_missingness_structure(data, contract, review_fraction=0.8, fail_fraction=0.2)
    with pytest.raises(GP3BayesError, match="condition-number"):
        ds.audit_fixed_effect_design(data, contract, condition_number_review=0.5)
    with pytest.raises(GP3BayesError, match="leverage_multiplier"):
        ds.audit_fixed_effect_design(data, contract, leverage_multiplier=0.5)
    with pytest.raises(GP3BayesError, match="positive integers"):
        ds.audit_random_effects_support(data, contract, minimum_repeated_rows=0)


def test_documentation_figure_generator_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    path = root / "dev" / "generate_doc_figures.py"
    spec = importlib.util.spec_from_file_location("gp3bayespy_doc_figures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.generate_all(tmp_path)
    assert len(manifest) >= 25
    assert all((tmp_path / item["file"]).exists() for item in manifest)


@pytest.fixture(autouse=True)
def _coverage_close_all_figures():
    yield
    plt.close("all")
