from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import gp3bayespy.specification_closure as sc
from gp3bayespy.exceptions import GP3BayesError

backends = importlib.import_module("gp3bayespy.backends")
postfit = importlib.import_module("gp3bayespy.postfit_exploration")
sensitivity = importlib.import_module("gp3bayespy.sensitivity")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_backend_capabilities_parity_and_schema_governance(tmp_path):
    caps = backends.backend_capabilities()
    assert set(caps["backend"]) == {"pymc", "cmdstanpy"}
    env = backends.validate_backend_environment("pymc", compile_test=False)
    assert env.status in {"ready", "pass", "fail"}
    assert not env.model_fitted

    left = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "mean": [0.1, 0.2],
            "sd": [0.5, 0.7],
            "mcse_mean": [0.02, 0.02],
        }
    )
    right = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "mean": [0.11, 0.19],
            "sd": [0.51, 0.69],
            "mcse_mean": [0.02, 0.02],
        }
    )
    audit = backends.audit_backend_parity(left, right)
    assert audit.status in {"pass", "review"}
    assert not audit.identical_draws_expected
    assert not audit.model_adequacy_established

    obj = {
        "family": "binary",
        "settings": {"chains": 2, "seed": 1},
        "table": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        "flags": [False, True],
    }
    schema = backends.capture_gp3bayes_schema(obj, max_depth=3)
    assert backends.compare_gp3bayes_schemas(schema, obj, compare_lengths=True).status == "pass"
    changed = {**obj, "settings": {"chains": 2}}
    assert (
        backends.compare_gp3bayes_schemas(schema, changed, compare_lengths=True).status == "review"
    )
    assert backends.validate_gp3bayes_schema(changed, schema).schema_compatibility_only
    with pytest.raises(GP3BayesError):
        backends.validate_gp3bayes_schema(changed, schema, strict=True)

    path = tmp_path / "schema.json"
    frozen = backends.freeze_gp3bayes_schema(schema, path)
    assert frozen.frozen
    assert backends.read_gp3bayes_schema(path).frozen
    with pytest.raises(GP3BayesError):
        backends.freeze_gp3bayes_schema(schema, path)
    with pytest.raises(GP3BayesError):
        backends.capture_gp3bayes_schema(obj, max_depth=-1)


def test_postfit_tables_and_diagnostics(monkeypatch):
    draws = pd.DataFrame(
        {
            "a": np.linspace(-1, 1, 200),
            "b": np.linspace(0, 2, 200),
            "c": np.sin(np.linspace(0, 3, 200)),
        }
    )
    assert len(postfit.posterior_interval_table(draws)) == 3
    assert "probability_in_rope" in postfit.posterior_probability_table(draws, rope=(-0.1, 0.1))
    assert len(postfit.posterior_correlation_table(draws, method="pearson")) == 3
    assert len(postfit.posterior_correlation_table(draws, method="spearman")) == 3

    loo = SimpleNamespace(
        pareto_k=np.array([0.2, 0.55, 0.8, 1.2]),
        elpd_loo=-12.0,
        se_elpd_loo=1.5,
        p_loo=2.4,
        looic=24.0,
    )
    assert postfit.loo_diagnostic_table(loo)["flagged"].sum() == 2
    assert len(postfit.loo_summary_table(loo)) == 3
    assert not postfit.model_comparison_table(
        pd.DataFrame({"elpd_diff": [0.0, -2.1], "se_diff": [0.0, 1.1]}, index=["m1", "m2"])
    )["automatic_selection"].any()
    assert not postfit.model_weights_table({"m1": 0.7, "m2": 0.3})["automatic_selection"].any()

    components = {
        "b_Intercept": np.array([[0.1, 0.2, 0.15, 0.18], [0.12, 0.19, 0.16, 0.17]]),
        "b_condition": np.array([[0.4, 0.5, 0.45, 0.48], [0.42, 0.49, 0.46, 0.47]]),
    }
    monkeypatch.setattr(postfit, "_posterior_components", lambda fit: components)
    fit = SimpleNamespace(fit_performed=True, family="binary")
    diag = postfit.mcmc_diagnostic_table(fit)
    issues = postfit.identify_mcmc_issues(diag, min_bulk_ess=2, min_tail_ess=2, max_mcse_fraction=1)
    assert len(diag) == 2 and len(issues) == 2

    variance = postfit.variance_component_table(
        {
            "sd_participant": np.linspace(0.1, 0.5, 100),
            "sigma": np.linspace(0.4, 0.8, 100),
        }
    )
    assert set(variance["variable"]) == {"sd_participant", "sigma"}

    with pytest.raises(GP3BayesError):
        postfit.posterior_correlation_table(draws, method="bad")
    with pytest.raises(GP3BayesError):
        postfit.posterior_interval_table(draws, variables="missing")


def _estimands():
    ref = sc.Estimand(
        "binary",
        "contrast",
        pd.DataFrame({".draw": np.arange(1, 101), "contrast": np.linspace(0.1, 0.3, 100)}),
        {},
    )
    alt = sc.Estimand(
        "binary",
        "contrast",
        pd.DataFrame({".draw": np.arange(1, 101), "contrast": np.linspace(0.12, 0.32, 100)}),
        {},
    )
    return ref, alt


def test_sensitivity_evidence_and_publication_graphics(tmp_path):
    ref, alt = _estimands()
    plan = sensitivity.create_sensitivity_suite_plan(alternative_estimands={"alternate": alt})
    fit = SimpleNamespace(family="binary")
    suite = sensitivity.run_sensitivity_suite(fit, plan, reference_estimand=ref)
    assert suite.status in {"completed", "review"}
    assert not sensitivity.summarise_sensitivity_suite(suite).empty

    evidence = sensitivity.collect_model_evidence(
        fit=fit,
        design=SimpleNamespace(status="pass"),
        posterior=SimpleNamespace(status="completed"),
        estimands=ref,
        sensitivity=suite,
    )
    report = sensitivity.create_model_evidence_report(evidence, tmp_path / "evidence.md")
    assert Path(report).is_file()

    cmp = sc.compare_estimand_sensitivity(ref, {"alternate": alt}).table
    recovery = pd.DataFrame(
        {
            "variable": ["a", "b"],
            "standardized_bias": [0.1, -0.2],
            "coverage": [0.9, 0.95],
            "rmse": [0.3, 0.4],
        }
    )
    estimates = pd.DataFrame(
        {
            "variable": ["a", "a", "b", "b"],
            "truth": [0, 0, 1, 1],
            "median": [0.1, -0.05, 0.9, 1.1],
            "lower": [-0.2, -0.3, 0.5, 0.7],
            "upper": [0.4, 0.2, 1.3, 1.5],
            "repetition": [1, 2, 1, 2],
        }
    )
    status = pd.DataFrame(
        {
            "diagnostic_status": ["pass", "review", "pass"],
            "completed": [True, True, False],
        }
    )
    prior = pd.DataFrame(
        {
            "scenario": ["s1", "s2"],
            "scale_multiplier": [0.5, 2.0],
            "variable": ["b", "b"],
            "standardized_shift": [0.1, 0.2],
        }
    )
    scenarios = pd.DataFrame(
        {
            "scenario": ["s1", "s2"],
            "maximum_standardized_shift": [0.1, 0.2],
        }
    )
    deletion = pd.DataFrame({"omitted_unit": ["p1", "p2"], "median_shift": [0.02, -0.03]})
    power = pd.DataFrame(
        {
            "variable": ["b1", "b2"],
            "prior": [0.1, 0.2],
            "likelihood": [0.05, 0.08],
        }
    )
    sbc_stats = pd.DataFrame(
        {
            "parameter": ["b", "b", "sd", "sd"],
            "rank": [10, 40, 20, 35],
            "draws": [50, 50, 50, 50],
            "coverage": [0.9, 0.9, 0.95, 0.95],
            "truth": [0.0, 0.1, 0.5, 0.6],
            "median": [0.02, 0.08, 0.48, 0.65],
        }
    )
    sbc = {
        "raw": {"stats": sbc_stats},
        "plan": {"family": "binary", "backend": "pymc", "n_sims": 4},
        "status": "completed",
        "diagnostics_inspected": True,
    }

    figs = (
        sensitivity.plot_recovery_bias(recovery),
        sensitivity.plot_recovery_coverage(recovery),
        sensitivity.plot_recovery_rmse(recovery),
        sensitivity.plot_recovery_estimates(estimates),
        sensitivity.plot_recovery_fit_status(status),
        sensitivity.plot_prior_sensitivity(prior),
        sensitivity.plot_prior_sensitivity_scenarios(scenarios),
        sensitivity.plot_estimand_sensitivity_gg(cmp),
        sensitivity.plot_group_deletion_sensitivity(deletion),
        sensitivity.plot_random_slope_sensitivity(cmp),
        sensitivity.plot_powerscale_sensitivity_gg(power),
        sensitivity.plot_sbc_rank_gg(sbc),
        sensitivity.plot_sbc_ecdf_gg(sbc),
        sensitivity.plot_sbc_coverage_gg(sbc),
        sensitivity.plot_sbc_simulated_vs_estimated_gg(sbc),
    )
    assert all(fig.axes for fig in figs)
    assert not sensitivity.sbc_overview_table(sbc).empty

    with pytest.raises(GP3BayesError):
        sensitivity.collect_model_evidence(fit=fit, compute=("bad",))
