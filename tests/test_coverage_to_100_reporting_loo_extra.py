from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.loo as loo
import gp3bayespy.reporting as rep
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fig(title: str = "x"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    return fig


def test_reporting_registry_inventory_dashboard_and_files(tmp_path, monkeypatch):
    f1 = _fig("one")
    f2 = _fig("two")

    fs = rep.create_figure_set({"one": f1, "two": f2}, title="set")
    assert fs.names == ("one", "two")
    saved = rep.save_figure_set(fs, tmp_path / "figs", dpi=72)
    assert len(saved) == 2
    with pytest.raises(GP3BayesError):
        rep.save_figure_set(fs, tmp_path / "figs", dpi=72)
    assert len(rep.save_figure_set(fs, tmp_path / "figs", dpi=72, overwrite=True)) == 2

    for bad in ({}, {"": f1}, {"x": object()}):
        with pytest.raises(GP3BayesError):
            rep.create_figure_set(bad)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        rep.save_figure_set(object(), tmp_path)  # type: ignore[arg-type]

    assert rep.theme_gp3bayes(10)["base_size"] == 10.0
    with pytest.raises(GP3BayesError):
        rep.theme_gp3bayes(0)

    reg = rep.create_publication_registry("paper")
    rep.register_publication_table(
        reg,
        "table1",
        pd.DataFrame({"a": [1]}),
        caption="A",
        source="test",
    )
    rep.register_publication_figure(reg, "figure1", f1, caption="F")
    assert len(rep.publication_registry_table(reg)) == 2
    assert rep.validate_publication_registry(reg).valid

    registry_path = tmp_path / "registry.md"
    assert registry_path.name in rep.write_publication_registry(reg, registry_path)
    with pytest.raises(GP3BayesError):
        rep.write_publication_registry(reg, registry_path)
    rep.write_publication_registry(reg, registry_path, overwrite=True)
    assert len(rep.save_publication_registry_figures(reg, tmp_path / "registry_figs")) == 1

    empty_reg = rep.create_publication_registry()
    assert rep.save_publication_registry_figures(empty_reg, tmp_path / "none").empty

    with pytest.raises(GP3BayesError):
        rep.create_publication_registry("")
    with pytest.raises(GP3BayesError):
        rep.register_publication_table(reg, "", pd.DataFrame())
    with pytest.raises(GP3BayesError):
        rep.register_publication_table(reg, "bad", object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        rep.register_publication_table(reg, "table1", pd.DataFrame())
    with pytest.raises(GP3BayesError):
        rep.register_publication_figure(reg, "", f2)
    with pytest.raises(GP3BayesError):
        rep.register_publication_figure(reg, "badfig", object())
    with pytest.raises(GP3BayesError):
        rep.register_publication_figure(reg, "figure1", f2)
    with pytest.raises(GP3BayesError):
        rep.publication_registry_table(object())  # type: ignore[arg-type]

    broken = rep.create_publication_registry("broken")
    broken.tables["orphan"] = pd.DataFrame({"x": [1]})
    assert not rep.validate_publication_registry(broken).valid
    with pytest.raises(GP3BayesError):
        rep.write_publication_registry(broken, tmp_path / "broken.md")

    inventory = rep.create_complete_evidence_inventory(
        {
            "none": None,
            "status": SimpleNamespace(status="review"),
            "mapping": {"status": "pass"},
            "plain": object(),
        },
        "inventory",
    )
    inv_table = rep.evidence_inventory_table(inventory)
    assert {"not_available", "review", "pass", "available"}.issubset(set(inv_table["status"]))
    with pytest.raises(GP3BayesError):
        rep.create_complete_evidence_inventory({})
    with pytest.raises(GP3BayesError):
        rep.evidence_inventory_table(object())  # type: ignore[arg-type]

    dashboard = rep.create_diagnostic_dashboard(
        fit=object(),
        model_card=object(),
        label="dashboard",
    )
    assert len(rep.diagnostic_dashboard_table(dashboard)) == 8
    assert rep.plot_diagnostic_dashboard(dashboard).axes

    dashboard_path = tmp_path / "dashboard.md"
    assert dashboard_path.name in rep.write_diagnostic_dashboard_report(dashboard, dashboard_path)
    with pytest.raises(GP3BayesError):
        rep.write_diagnostic_dashboard_report(dashboard, dashboard_path)
    rep.write_diagnostic_dashboard_report(dashboard, dashboard_path, overwrite=True)

    with pytest.raises(GP3BayesError):
        rep.create_diagnostic_dashboard()
    with pytest.raises(GP3BayesError):
        rep.diagnostic_dashboard_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        rep.write_diagnostic_dashboard_report(object(), tmp_path / "x.md")  # type: ignore[arg-type]

    monkeypatch.setattr(
        rep,
        "plot_posterior_intervals",
        lambda *args, **kwargs: _fig("post"),
    )
    monkeypatch.setattr(
        rep,
        "plot_reporting_checklist",
        lambda *args, **kwargs: _fig("check"),
    )
    figs = rep.create_diagnostic_dashboard_figures(dashboard)
    assert set(figs.names) == {"posterior_intervals", "reporting_checklist"}

    only_bundle = rep.create_diagnostic_dashboard(analysis_bundle=object())
    with pytest.raises(GP3BayesError):
        rep.create_diagnostic_dashboard_figures(only_bundle)
    with pytest.raises(GP3BayesError):
        rep.create_diagnostic_dashboard_figures(object())  # type: ignore[arg-type]


def test_model_card_and_reporting_plot_matrix(tmp_path, monkeypatch):
    import gp3bayespy.postfit_exploration as postfit
    import gp3bayespy.unified_workflow_api as unified

    spec = SimpleNamespace(
        model_family="hierarchical_binary",
        formula_text="y ~ x",
    )
    fit = SimpleNamespace(
        family="binary",
        specification=spec,
        sampling_backend="pymc",
        sampling={"chains": 2},
        package_versions={"gp3bayespy": "0.5.0"},
    )

    monkeypatch.setattr(
        unified,
        "diagnose_model_fit",
        lambda x: SimpleNamespace(status="pass"),
    )
    monkeypatch.setattr(
        unified,
        "model_workflow_status",
        lambda x: SimpleNamespace(status="review"),
    )
    card = rep.create_model_card(
        fit,
        analysis_bundle=SimpleNamespace(status="pass"),
        manifest={"status": "available"},
        label="m",
    )
    assert len(rep.model_card_table(card)) == 5
    checklist = rep.create_reporting_checklist(card)
    assert len(checklist) == 9
    assert rep.plot_reporting_checklist(checklist).axes

    card_path = tmp_path / "model-card.md"
    assert card_path.name in rep.write_model_card(card, card_path)
    with pytest.raises(GP3BayesError):
        rep.write_model_card(card, card_path)
    rep.write_model_card(card, card_path, overwrite=True)

    with pytest.raises(GP3BayesError):
        rep.create_model_card(SimpleNamespace(family="pupil"))
    with pytest.raises(GP3BayesError):
        rep.model_card_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        rep.write_model_card(object(), tmp_path / "x")  # type: ignore[arg-type]

    def boom(_):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(unified, "diagnose_model_fit", boom)
    monkeypatch.setattr(unified, "model_workflow_status", boom)
    failed_card = rep.create_model_card(fit)
    failed_checks = rep.create_reporting_checklist(failed_card)
    assert not failed_checks.loc[
        failed_checks["item"] == "diagnostics_available",
        "available",
    ].iloc[0]

    draws = pd.DataFrame(
        {
            "a": np.linspace(-1, 1, 60),
            "b": np.linspace(1, -1, 60) + np.sin(np.arange(60)) * 0.1,
        }
    )
    interval = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "lower": [-1.0, -0.5],
            "median": [0.0, 0.2],
            "upper": [1.0, 0.9],
        }
    )
    assert rep.plot_posterior_intervals(interval).axes
    assert rep.plot_posterior_areas(interval).axes
    assert rep.plot_posterior_density(draws).axes
    assert rep.plot_posterior_correlations(draws).axes
    assert rep.plot_posterior_correlations(draws.to_numpy()).axes

    monkeypatch.setattr(postfit, "extract_posterior_draws", lambda *a, **k: draws)
    assert rep.plot_posterior_pairs(object(), max_variables=2).axes
    assert rep.plot_rank_diagnostics(object()).axes
    assert rep.plot_autocorrelation(object(), lags=5).axes

    corr_table = pd.DataFrame(
        {
            "variable1": ["a"],
            "variable2": ["b"],
            "correlation": [0.3],
        }
    )
    monkeypatch.setattr(
        postfit,
        "posterior_correlation_table",
        lambda *a, **k: corr_table,
    )
    assert rep.plot_posterior_correlations(object()).axes

    assert rep.plot_mcmc_quality(pd.DataFrame({"status": ["pass", "review"]})).axes
    assert rep.plot_mcmc_quality(pd.DataFrame({"value": [1, 2]})).axes

    monkeypatch.setattr(
        postfit,
        "sampler_diagnostic_table",
        lambda fit: pd.DataFrame({"energy": [1.0, 2.0], "depth": [3, 4]}),
    )
    assert rep.plot_sampler_diagnostics(object()).axes
    monkeypatch.setattr(
        postfit,
        "sampler_diagnostic_table",
        lambda fit: pd.DataFrame({"note": ["x", "y"]}),
    )
    assert rep.plot_sampler_diagnostics(object()).axes

    estimand = SimpleNamespace(
        draws={
            "effect": np.linspace(-1, 1, 50),
            "other": np.linspace(0, 2, 50),
        }
    )
    assert rep.plot_estimand_intervals(estimand, quantities=("effect",)).axes
    assert rep.plot_estimand_intervals(interval).axes

    pred = SimpleNamespace(
        summary=pd.DataFrame(
            {
                "predicted_mean": [0.2, 0.8],
                "lower": [0.1, 0.6],
                "upper": [0.4, 0.9],
            }
        )
    )
    assert rep.plot_prediction_intervals(pred).axes
    assert rep.plot_prediction_intervals(pd.DataFrame({"mean": [1.0, 2.0]})).axes

    calibration = pd.DataFrame(
        {
            "mean_predicted_probability": [0.2, 0.8],
            "observed_rate": [0.1, 0.9],
        }
    )
    assert rep.plot_binary_calibration(calibration).axes

    thresholds = pd.DataFrame(
        {
            "threshold": [0.3, 0.5],
            "accuracy": [0.7, 0.8],
            "sensitivity": [0.8, 0.7],
            "specificity": [0.6, 0.9],
            "balanced_accuracy": [0.7, 0.8],
        }
    )
    assert rep.plot_binary_threshold_metrics(thresholds).axes

    quantile = SimpleNamespace(
        table=pd.DataFrame(
            {
                "probability": [0.1, 0.5, 0.9],
                "empirical_probability": [0.15, 0.55, 0.85],
            }
        )
    )
    assert rep.plot_duration_quantile_calibration(quantile).axes
    assert rep.plot_duration_pit(SimpleNamespace(table=pd.DataFrame({"pit": [0.1, 0.5, 0.9]}))).axes
    assert rep.plot_exceedance_probability(pd.DataFrame({"probability": [0.2, 0.7]})).axes
    assert rep.plot_predictive_coverage(
        pd.DataFrame(
            {
                "nominal_coverage": [0.5, 0.9],
                "empirical_coverage": [0.55, 0.88],
            }
        )
    ).axes
    assert rep.plot_predictive_residuals(pd.DataFrame({"residual": [-0.2, 0.3]})).axes
    assert rep.plot_prediction_support(
        SimpleNamespace(table=pd.DataFrame({"status": ["inside", "outside"]}))
    ).axes
    assert rep.plot_prediction_support(SimpleNamespace(table=pd.DataFrame({"detail": ["x"]}))).axes

    assert rep.plot_uncertainty_decomposition(
        SimpleNamespace(
            table=pd.DataFrame(
                {
                    "epistemic_variance": [1.0, 2.0],
                    "residual_variance": [2.0, 1.0],
                    "total_variance": [3.0, 3.0],
                }
            )
        )
    ).axes
    assert (
        rep.plot_uncertainty_decomposition(SimpleNamespace(table=pd.DataFrame({"x": [1]}))) is None
    )

    assert rep.plot_grouped_prediction_check(
        SimpleNamespace(
            table=pd.DataFrame(
                {
                    "group": ["a", "b"],
                    "observed": [1.0, 2.0],
                    "predicted_mean": [1.1, 1.9],
                }
            )
        )
    ).axes
    assert rep.plot_group_effects(
        pd.DataFrame(
            {
                "group": ["g", "h"],
                "level": ["a", "b"],
                "median": [0.1, 0.2],
            }
        ),
        groups=("g",),
    ).axes
    assert rep.plot_group_effects(pd.DataFrame({"label": ["a"]})).axes
    assert rep.plot_variance_components(pd.DataFrame({"component": ["p"], "variance": [0.2]})).axes
    assert rep.plot_variance_components(pd.DataFrame({"component": ["p"]})).axes

    with pytest.raises(GP3BayesError):
        rep._df(object(), "table")
    assert rep._status(None) == "not_available"
    assert rep._status({"status": "warn"}) == "warn"
    assert rep._status(object()) == "available"


def test_loo_pointwise_group_atlas_and_plot_paths():
    x = SimpleNamespace(
        pointwise=np.array(
            [
                [-1.0, 0.1],
                [-2.0, 0.2],
                [-3.0, 0.3],
                [-4.0, 0.4],
            ]
        ),
        pareto_k=np.array([0.2, 0.8, 1.2, np.nan]),
        influence_pareto_k=np.array([0.1, 0.7, 1.1, 0.9]),
    )
    data = pd.DataFrame({"group": ["a", "a", "b", "b"]})
    point = loo.loo_pointwise_table(x, data=data)
    assert list(point["flagged"]) == [False, True, True, True]
    assert point["severe"].sum() == 1

    one_d = SimpleNamespace(
        pointwise=np.array([-1.0, -2.0]),
        pareto_k=np.array([0.2, 0.3]),
    )
    assert len(loo.loo_pointwise_table(one_d)) == 2

    frame_x = SimpleNamespace(
        pointwise=pd.DataFrame({"elpd_loo": [-1.0, -2.0], "pareto_k": [0.4, 0.9]})
    )
    assert len(loo.loo_pointwise_table(frame_x)) == 2

    raw = SimpleNamespace(
        raw=SimpleNamespace(pointwise=np.array([-1.0, -2.0])),
        pareto_k=np.array([0.1, 0.2]),
    )
    assert len(loo.loo_pointwise_table(raw)) == 2

    summary = loo.loo_influence_summary(point)
    assert summary.loc[0, "observations"] == 4
    flagged = loo.loo_flagged_data(point, threshold=0.7)
    assert len(flagged) == 2
    grouped = loo.loo_group_influence_table(point, "group")
    assert len(grouped) == 2

    atlas = loo.create_loo_influence_atlas(x, data=data, threshold=0.7)
    assert len(atlas.to_frame()) == 4
    assert len(loo.loo_influence_atlas_table(atlas)) == 4
    assert len(loo._table(atlas)) == 4

    figures = [
        loo.plot_loo_pointwise_elpd(atlas),
        loo.plot_loo_pareto_vs_elpd(atlas),
        loo.plot_loo_influence_rank(atlas),
        loo.plot_loo_group_influence(grouped),
        loo.plot_loo_group_elpd(grouped),
    ]
    assert all(fig.axes for fig in figures)

    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(object())
    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(
            SimpleNamespace(
                pointwise=np.ones((2, 2, 2)),
                pareto_k=np.ones(2),
            )
        )
    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(SimpleNamespace(pointwise=np.array([1.0])))
    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(
            SimpleNamespace(
                pointwise=np.array([1.0, 2.0]),
                pareto_k=np.array([0.1]),
            )
        )
    with pytest.raises(GP3BayesError):
        loo.loo_pointwise_table(x, data=pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        loo.loo_flagged_data(point, threshold=np.inf)
    with pytest.raises(GP3BayesError):
        loo.loo_group_influence_table(point, "missing")
    with pytest.raises(GP3BayesError):
        loo.loo_influence_atlas_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        loo.plot_loo_pointwise_elpd(pd.DataFrame({"pareto_k": [0.2]}))
    with pytest.raises(GP3BayesError):
        loo.plot_loo_pareto_vs_elpd(pd.DataFrame({"pareto_k": [0.2]}))
    with pytest.raises(GP3BayesError):
        loo.plot_loo_group_influence(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        loo.plot_loo_group_elpd(pd.DataFrame({"x": [1]}))
    with pytest.raises(GP3BayesError):
        loo.plot_loo_group_elpd(pd.DataFrame({"group_value": ["a"], "total_elpd_loo": [np.nan]}))

    all_nan = pd.DataFrame(
        {
            "observation": [1, 2],
            "pareto_k": [np.nan, np.nan],
            "influence_pareto_k": [np.nan, np.nan],
        }
    )
    assert np.isnan(loo.loo_influence_summary(all_nan).loc[0, "median_pareto_k"])
    nan_group = all_nan.assign(group=["a", "a"])
    grouped_nan = loo.loo_group_influence_table(nan_group, "group")
    assert np.isnan(grouped_nan.loc[0, "mean_pareto_k"])
