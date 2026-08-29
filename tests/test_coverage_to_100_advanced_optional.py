from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.advanced_optional_workflows as aow
from gp3bayespy.exceptions import BackendUnavailableError, GP3BayesError


def _binary_spec(interaction: bool = True):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=6, trials_per_participant=4, n_items=4, seed=901
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate") if interaction else None,
        random_slope=True,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    return prepared, gp.specify_binary_model(prepared)


def _duration_spec(interaction: bool = True):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=6, trials_per_participant=4, n_items=4, seed=902
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate") if interaction else None,
        random_slope=True,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    return prepared, gp.specify_duration_model(prepared, baseline=500.0)


def test_interaction_prior_specification_summary_translation_and_validation(monkeypatch):
    bp, _ = _binary_spec()
    dp, _ = _duration_spec()

    b = aow.specify_binary_model_with_interaction_prior(
        bp, baseline=0.5, main_effect_scale=0.8, interaction_scale=0.4
    )
    d = aow.specify_duration_model_with_interaction_prior(
        dp, baseline=500.0, main_effect_scale=0.4, interaction_scale=0.2
    )
    assert b.family == "binary"
    assert d.family == "duration"
    assert aow.interaction_prior_summary(b).loc[0, "interaction_scale"] == 0.4
    assert aow.interaction_prior_summary(d).loc[0, "interaction_scale"] == 0.2

    monkeypatch.setattr(
        aow, "translate_binary_model_to_brms", lambda spec: SimpleNamespace(family="binary")
    )
    monkeypatch.setattr(
        aow, "translate_duration_model_to_brms", lambda spec: SimpleNamespace(family="duration")
    )
    assert aow.translate_binary_model_with_interaction_prior(b).interaction_scale == 0.4
    assert aow.translate_duration_model_with_interaction_prior(d).interaction_scale == 0.2

    bp_no, _ = _binary_spec(interaction=False)
    with pytest.raises(GP3BayesError, match="interaction"):
        aow.specify_binary_model_with_interaction_prior(bp_no, baseline=0.5)
    with pytest.raises(GP3BayesError, match="positive"):
        aow.specify_binary_model_with_interaction_prior(bp, baseline=0.5, main_effect_scale=0)
    with pytest.raises(GP3BayesError, match="positive"):
        aow.specify_duration_model_with_interaction_prior(dp, baseline=500.0, interaction_scale=-1)
    with pytest.raises(GP3BayesError):
        aow.interaction_prior_summary(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        aow.translate_binary_model_with_interaction_prior(d)
    with pytest.raises(GP3BayesError):
        aow.translate_duration_model_with_interaction_prior(b)


def test_backend_aliases_cmdstan_guard_and_invalid_backend(monkeypatch):
    _, bspec = _binary_spec()
    _, dspec = _duration_spec()

    monkeypatch.setattr(aow, "fit_binary_model", lambda *a, **k: ("binary", k["seed"]))
    monkeypatch.setattr(aow, "fit_duration_model", lambda *a, **k: ("duration", k["seed"]))

    assert aow.fit_binary_model_backend(bspec, backend="pymc", seed=7) == ("binary", 7)
    assert aow.fit_binary_model_backend(bspec, backend="rstan", seed=8) == ("binary", 8)
    assert aow.fit_duration_model_backend(dspec, backend="pymc", seed=9) == ("duration", 9)
    assert aow.fit_duration_model_backend(dspec, backend="rstan", seed=10) == ("duration", 10)

    monkeypatch.setattr(aow, "check_cmdstan_backend", lambda strict=False: None)
    with pytest.raises(BackendUnavailableError):
        aow.fit_binary_model_backend(bspec, backend="cmdstanr")
    with pytest.raises(BackendUnavailableError):
        aow.fit_duration_model_backend(dspec, backend="cmdstanpy")
    with pytest.raises(GP3BayesError, match="backend"):
        aow.fit_binary_model_backend(bspec, backend="unknown")
    with pytest.raises(GP3BayesError, match="backend"):
        aow.fit_duration_model_backend(dspec, backend="unknown")

    with pytest.raises(BackendUnavailableError):
        aow.fit_binary_model_cmdstanr(bspec)
    with pytest.raises(BackendUnavailableError):
        aow.fit_duration_model_cmdstanr(dspec)


def test_psis_smoothing_loo_comparison_weights_and_influence(monkeypatch):
    tiny, k_tiny = aow._psis_smooth(np.array([0.0, 0.2, -0.1, 0.4]))
    assert np.isclose(tiny.sum(), 1.0)
    assert np.isnan(k_tiny)

    weights, _ = aow._psis_smooth(np.linspace(-1.0, 1.0, 80))
    assert np.isclose(weights.sum(), 1.0)

    with pytest.raises(GP3BayesError):
        aow._psis_smooth(np.array([[1.0, 2.0]]))
    with pytest.raises(GP3BayesError):
        aow._psis_smooth(np.array([0.0, np.nan]))

    rng = np.random.default_rng(903)
    ll1 = rng.normal(-1.0, 0.25, size=(120, 12))
    ll2 = rng.normal(-1.03, 0.30, size=(120, 12))
    loo1 = aow.compute_psis_loo_from_log_lik(ll1)
    loo2 = aow.compute_psis_loo_from_log_lik(ll2, chain_id=np.repeat([1, 2], 60))
    one = aow.compute_psis_loo_from_log_lik(ll1[:, :1])
    assert len(loo1.pointwise) == 12
    assert one.se_elpd_loo == 0.0

    with pytest.raises(GP3BayesError, match="finite"):
        aow.compute_psis_loo_from_log_lik(np.array([1.0, 2.0]))
    with pytest.raises(GP3BayesError, match="chain_id"):
        aow.compute_psis_loo_from_log_lik(ll1, chain_id=[1, 2])

    infl = aow.identify_loo_influential_observations(loo1, threshold=-999)
    assert len(infl) == 12
    data = pd.DataFrame({"row_id": np.arange(12)})
    assert "row_id" in aow.identify_loo_influential_observations(loo1, threshold=-999, data=data)
    with pytest.raises(GP3BayesError):
        aow.identify_loo_influential_observations(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="one row"):
        aow.identify_loo_influential_observations(loo1, threshold=-999, data=data.iloc[:-1])

    comparison = aow.compare_psis_loo({"m1": loo1, "m2": loo2})
    assert len(comparison.comparison) == 2
    pseudo = aow.compute_loo_model_weights(comparison, method="pseudobma")
    stacking = aow.compute_loo_model_weights(comparison, method="stacking")
    assert np.isclose(sum(pseudo.weights.values()), 1.0)
    assert np.isclose(sum(stacking.weights.values()), 1.0)

    with pytest.raises(GP3BayesError, match="at least two"):
        aow.compare_psis_loo({"m1": loo1})
    mismatched = aow.compute_psis_loo_from_log_lik(ll2[:, :10])
    with pytest.raises(GP3BayesError, match="same number"):
        aow.compare_psis_loo({"m1": loo1, "m2": mismatched})
    with pytest.raises(GP3BayesError, match="method"):
        aow.compute_loo_model_weights(comparison, method="bad")
    with pytest.raises(GP3BayesError, match="at least two"):
        aow.compute_loo_model_weights({"m1": loo1})


def test_psis_fit_wrapper_and_explicit_refit_guard(monkeypatch):
    rng = np.random.default_rng(904)
    ll = rng.normal(-1.0, 0.2, size=(60, 5))
    monkeypatch.setattr(aow, "extract_log_likelihood", lambda fit: ll)
    fit = SimpleNamespace()
    result = aow.compute_psis_loo(fit)
    assert result.source is fit
    with pytest.raises(GP3BayesError, match="moment_match"):
        aow.compute_psis_loo(fit, moment_match=True)
    with pytest.raises(GP3BayesError, match="reloo"):
        aow.compute_psis_loo(fit, reloo=True)


def test_pathological_simulations_cover_every_declared_scenario():
    binary_scenarios = (
        "null_contrast",
        "weak_information",
        "severe_imbalance",
        "near_separation",
        "omitted_random_slope",
        "sparse_item_structure",
        "all_zero_participants",
        "rank_deficiency",
        "missing_outcomes",
    )
    duration_scenarios = (
        "high_group_heterogeneity",
        "weak_information",
        "severe_imbalance",
        "heavy_tailed_contamination",
        "mixture",
        "censoring",
        "incorrect_unit",
        "zero_duration",
        "negative_duration",
        "null_ratio",
    )

    for i, scenario in enumerate(binary_scenarios):
        sim = aow.simulate_binary_pathology(scenario, seed=1000 + i)
        table = aow.evaluate_pathological_simulation(sim)
        assert table.loc[0, "scenario"] == scenario

    for i, scenario in enumerate(duration_scenarios):
        sim = aow.simulate_duration_pathology(scenario, seed=1100 + i)
        table = aow.evaluate_pathological_simulation(sim)
        assert table.loc[0, "scenario"] == scenario

    with pytest.raises(GP3BayesError, match="Unknown binary"):
        aow.simulate_binary_pathology("unknown")
    with pytest.raises(GP3BayesError, match="Unknown duration"):
        aow.simulate_duration_pathology("unknown")
    with pytest.raises(GP3BayesError):
        aow.evaluate_pathological_simulation(object())  # type: ignore[arg-type]


def test_sbc_custom_and_brms_plans_summary_and_powerscale():
    def generator(seed: int):
        rng = np.random.default_rng(seed)
        return {
            "truth": {"theta": 0.25},
            "draws": {"theta": rng.normal(0.25, 0.3, 80)},
        }

    plan = aow.create_custom_sbc_plan(generator, "pymc", n_sims=4, seed=12)
    result = aow.run_sbc_plan(plan)
    summary = aow.summarise_sbc_result(result)
    assert len(result.simulations) == 4
    assert len(result.ranks) == 4
    assert summary["overview"].loc[0, "calibration_assessed"]

    empty_plan = aow.create_custom_sbc_plan(
        lambda seed: pd.DataFrame({"x": [seed]}), "pymc", n_sims=2, seed=13
    )
    empty = aow.run_sbc_plan(empty_plan)
    empty_summary = aow.summarise_sbc_result(empty)
    assert not empty_summary["overview"].loc[0, "calibration_assessed"]

    _, bspec = _binary_spec()
    _, dspec = _duration_spec()
    assert aow.create_brms_sbc_plan(bspec, n_sims=2).generator_args["scenario"] == "null_contrast"
    assert aow.create_brms_sbc_plan(dspec, n_sims=2).generator_args["scenario"] == "null_ratio"

    with pytest.raises(GP3BayesError):
        aow.create_custom_sbc_plan(None, "pymc")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        aow.create_custom_sbc_plan(generator, "pymc", n_sims=0)
    with pytest.raises(GP3BayesError):
        aow.create_brms_sbc_plan(SimpleNamespace(family="other"))
    with pytest.raises(GP3BayesError):
        aow.run_sbc_plan(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        aow.summarise_sbc_result(object())  # type: ignore[arg-type]

    both = aow.powerscale_sequence_for_fit(object(), variable="b")
    prior = aow.powerscale_sequence_for_fit(object(), component="prior")
    assessed = aow.assess_powerscaled_sensitivity(object(), variable="b")
    assert set(both["component"]) == {"prior", "likelihood"}
    assert set(prior["component"]) == {"prior"}
    assert not assessed["robustness_established"].any()
    with pytest.raises(GP3BayesError, match="component"):
        aow.powerscale_sequence_for_fit(object(), component="bad")


def test_binary_separation_screening_and_input_errors():
    prepared, spec = _binary_spec()
    result = aow.detect_binary_separation(spec)
    assert result["status"] in {"pass", "review"}
    assert result["automatic_variable_removal"] is False

    with pytest.raises(GP3BayesError, match="data and a contract"):
        aow.detect_binary_separation(object())

    bad_contract = SimpleNamespace(mappings={"outcome": "missing"})
    bad_spec = SimpleNamespace(
        prepared=SimpleNamespace(data=prepared.data),
        contract=bad_contract,
    )
    with pytest.raises(GP3BayesError, match="outcome"):
        aow.detect_binary_separation(bad_spec)
