from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.specification_closure as sc
from gp3bayespy.exceptions import GP3BayesError


def _binary():
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=6, trials_per_participant=4, n_items=4, seed=101
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=True,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_binary_model(prepared)
    return sim.data, contract, prepared, spec


def _duration():
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=6, trials_per_participant=4, n_items=4, seed=102
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("condition", "participant_covariate"),
        random_slope=True,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    return sim.data, contract, prepared, spec


def test_closure_design_audits_binary_and_duration_paths():
    bdata, bcontract, bprep, _ = _binary()
    ddata, dcontract, dprep, _ = _duration()

    balance = sc.summarise_condition_balance(bdata, bcontract)
    assert balance.status in {"pass", "review"}
    variation_p = sc.summarise_binary_group_variation(bdata, bcontract, "participant")
    variation_i = sc.summarise_binary_group_variation(bdata, bcontract, "item")
    assert len(variation_p.table) and len(variation_i.table)

    ident = sc.identify_identifier_like_predictors(bdata, bcontract)
    assert set(ident.table["predictor"]) == set(bcontract.predictors)

    extremes = sc.review_duration_extremes(ddata, dcontract)
    assert len(extremes.table) == len(ddata)

    bounded = ddata.copy()
    bounded["censor"] = [False] * len(bounded)
    boundary = sc.audit_duration_boundaries(
        bounded,
        dcontract,
        allowed_range=(
            float(bounded["duration"].min()) * 0.9,
            float(bounded["duration"].max()) * 1.1,
        ),
        censor_col="censor",
    )
    assert boundary.status in {"pass", "review"}

    strict_b = sc.audit_model_readiness_strict(bprep.data, bcontract, run_separation=False)
    strict_d = sc.audit_model_readiness_strict(dprep.data, dcontract, run_separation=False)
    assert strict_b.status in {"ready", "ready_with_warnings", "not_ready"}
    assert strict_d.status in {"ready", "ready_with_warnings", "not_ready"}

    with pytest.raises(GP3BayesError, match="participant or item"):
        sc.summarise_binary_group_variation(bdata, bcontract, "bad")
    with pytest.raises(GP3BayesError, match="positive"):
        bad = ddata.copy()
        bad.loc[bad.index[0], "duration"] = 0
        sc.review_duration_extremes(bad, dcontract)


def test_transformation_recipe_replay_binary_and_duration():
    bdata, _, bprep, _ = _binary()
    ddata, _, dprep, _ = _duration()

    for prepared in (bprep, dprep):
        recipe = sc.create_transformation_recipe(prepared)
        raw = sc.invert_transformation_recipe(prepared.data, recipe)
        replay = sc.apply_transformation_recipe(
            raw,
            recipe,
            require_outcome=True,
            input_unit=recipe.transformations.get("outcome", {}).get("source_unit")
            if recipe.family == "duration"
            else None,
        )
        assert list(replay.columns) == list(prepared.data.columns)
        audit = sc.validate_transformation_replay(prepared)
        assert audit.replay_established
        already = sc.apply_transformation_recipe(prepared.data, recipe, input_scale="prepared")
        assert len(already) == len(prepared.data)

    with pytest.raises(GP3BayesError, match="input_scale"):
        sc.apply_transformation_recipe(
            bdata.head(), sc.create_transformation_recipe(bprep), input_scale="bad"
        )


def test_sensitivity_specification_builders_and_plans_are_bounded():
    _, _, bprep, bspec = _binary()
    _, _, dprep, dspec = _duration()

    slope_plan = sc.create_random_slope_sensitivity_plan(bspec)
    assert slope_plan.automatic_selection is False

    participant_col = str(bspec.contract.mappings["participant"])
    available = tuple(pd.unique(bspec.prepared.data[participant_col].astype(str)))
    deletion = sc.create_group_deletion_sensitivity_plan(
        bspec, group="participant", units=available[:2], max_units=4
    )
    assert deletion.automatic_exclusion is False
    assert len(deletion.table) == 2

    contrast = sc.create_contrast_coding_sensitivity_specification(
        bspec, condition_coding=(-1, 1), baseline=0.5
    )
    assert contrast.family == "binary"

    scaled = sc.create_predictor_scaling_sensitivity_specification(
        bspec,
        predictor="participant_covariate",
        scale_factor=2.0,
        coefficient_scale=0.75,
    )
    assert scaled.family == "binary"

    converted = sc.create_duration_unit_sensitivity_specification(
        dspec, multiplier=0.001, new_unit="seconds"
    )
    assert converted.family == "duration"

    with pytest.raises(GP3BayesError, match="participant or item"):
        sc.create_group_deletion_sensitivity_plan(bspec, group="bad")
    with pytest.raises(GP3BayesError, match="distinct"):
        sc.create_contrast_coding_sensitivity_specification(bspec, (1, 1), 0.5)


def test_estimand_summaries_sensitivity_and_duration_unit_invariance():
    ref_draws = pd.DataFrame(
        {
            ".draw": np.arange(1, 101),
            "contrast": np.linspace(0.1, 0.3, 100),
        }
    )
    alt_draws = pd.DataFrame(
        {
            ".draw": np.arange(1, 101),
            "contrast": np.linspace(0.11, 0.31, 100),
        }
    )
    ref = sc.Estimand("binary", "contrast", ref_draws, {"source": "test"})
    alt = sc.Estimand("binary", "contrast", alt_draws, {"source": "test"})
    summary = sc.summarise_estimand_draws(ref)
    assert summary.loc[0, "quantity"] == "contrast"
    comparison = sc.compare_estimand_sensitivity(ref, {"alternate": alt})
    assert comparison.status == "review"
    invariant = sc.audit_estimand_invariance(ref, alt, tolerance=0.02)
    assert invariant.invariance_established

    n = 100
    duration_ref = sc.Estimand(
        "duration",
        "conditional_median_ratio",
        pd.DataFrame(
            {
                ".draw": np.arange(1, n + 1),
                "conditional_median_ratio": np.repeat(1.2, n),
                "predictive_quantile_ratio": np.repeat(1.3, n),
                "reference_average_conditional_median": np.repeat(500.0, n),
                "focal_average_conditional_median": np.repeat(600.0, n),
                "reference_predictive_quantile": np.repeat(800.0, n),
                "focal_predictive_quantile": np.repeat(900.0, n),
            }
        ),
        {},
    )
    duration_converted = sc.Estimand(
        "duration",
        "conditional_median_ratio",
        duration_ref.draws.assign(
            reference_average_conditional_median=0.5,
            focal_average_conditional_median=0.6,
            reference_predictive_quantile=0.8,
            focal_predictive_quantile=0.9,
        ),
        {},
    )
    unit = sc.audit_duration_unit_invariance(
        duration_ref, duration_converted, multiplier=0.001, tolerance=1e-12
    )
    assert unit.invariance_established

    with pytest.raises(GP3BayesError, match="Unknown estimand quantities"):
        sc.summarise_estimand_draws(ref, quantities="missing")
    with pytest.raises(GP3BayesError, match="non-empty"):
        sc.compare_estimand_sensitivity(ref, {})


def test_kfold_and_traceability_with_backend_free_stub(monkeypatch):
    import importlib

    trace = sc.gp3bayes_specification_traceability()
    assert isinstance(trace, pd.DataFrame)
    assert len(trace) > 0
    assert not trace["automatic_decision"].any()

    advanced = importlib.import_module("gp3bayespy.advanced_optional_workflows")

    class FakeLOO:
        def __init__(self, n):
            self.pointwise = pd.DataFrame({"elpd_loo": np.repeat(-1.0, n)})

    monkeypatch.setattr(advanced, "compute_psis_loo", lambda fit: FakeLOO(12))
    fit = SimpleNamespace(family="binary", fit_performed=True)
    result = sc.compute_kfold_cv(fit, K=3, folds="random", seed=5)
    assert result.K == 3
    assert result.folds == "random"
    assert len(result.table) == 3
    assert result.automatic_selection is False
