from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.design_support_diagnostics as ds
import gp3bayespy.duration as duration
import gp3bayespy.predictive as predictive
import gp3bayespy.pupil as pupil
import gp3bayespy.specification_closure as c
import gp3bayespy.unified_workflow_api as u
from gp3bayespy.exceptions import GP3BayesError


def _binary_objects(seed: int = 2101):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=6,
        trials_per_participant=8,
        n_items=4,
        seed=seed,
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
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_binary_model(prepared)
    fit = SimpleNamespace(
        family="binary",
        fit_performed=True,
        specification=SimpleNamespace(prepared=prepared),
    )
    return sim, contract, prepared, spec, fit


def _duration_objects(seed: int = 2102):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=6,
        trials_per_participant=8,
        n_items=4,
        seed=seed,
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
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("participant_covariate", "trial_covariate"),
    )
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    fit = SimpleNamespace(
        family="duration",
        fit_performed=True,
        specification=SimpleNamespace(prepared=prepared),
        outcome_unit="milliseconds",
    )
    return sim, contract, prepared, spec, fit


def _fake_predict(fit, newdata=None, type="expected", ndraws=None, seed=1, **kwargs):
    data = fit.specification.prepared.data if newdata is None else newdata
    n = len(data)
    d = int(ndraws or 24)
    cond_col = fit.specification.prepared.contract.mappings.get("condition")
    cond = (
        pd.to_numeric(data[cond_col], errors="coerce").to_numpy(float)
        if cond_col in data
        else np.zeros(n)
    )
    offset = np.linspace(-0.01, 0.01, d)[:, None]
    if fit.family == "binary":
        base = np.clip(0.5 + 0.12 * cond, 0.05, 0.95)
        draws = np.clip(base[None, :] + offset, 0.01, 0.99)
    elif type == "linear":
        base = np.log(np.maximum(500.0 + 60.0 * cond, 10.0))
        draws = base[None, :] + offset
    else:
        base = np.maximum(500.0 + 60.0 * cond, 10.0)
        draws = np.maximum(base[None, :] + 10.0 * offset, 1.0)
    return SimpleNamespace(draws=draws)


def test_standardized_estimands_summaries_and_invariance(monkeypatch):
    monkeypatch.setattr(predictive, "predict_model", _fake_predict)

    bsim, _, bprepared, _, bfit = _binary_objects()
    b_est = c.estimate_standardized_probability_contrast(
        bfit,
        ndraws=20,
    )
    assert b_est.primary_quantity == "probability_difference"
    assert {"probability_difference", "probability_ratio"}.issubset(b_est.draws.columns)

    raw_b = c.invert_transformation_recipe(
        bprepared.data,
        bprepared,
    )
    b_raw = c.estimate_standardized_probability_contrast(
        bfit,
        target_data=raw_b.head(12),
        target_scale="raw",
        ndraws=20,
    )
    assert b_raw.metadata["target_rows"] == 12

    with pytest.raises(GP3BayesError):
        c.estimate_standardized_probability_contrast(
            SimpleNamespace(family="duration", fit_performed=True)
        )
    with pytest.raises(GP3BayesError):
        c._target_data(bfit, raw_b, "bad")

    _, _, dprepared, _, dfit = _duration_objects()
    d_est = c.estimate_standardized_duration_estimands(
        dfit,
        predictive_quantile=0.8,
        ndraws=20,
        seed=7,
    )
    assert d_est.primary_quantity == "conditional_median_ratio"
    assert "predictive_quantile_ratio" in d_est.draws

    raw_d = c.invert_transformation_recipe(dprepared.data, dprepared)
    d_raw = c.estimate_standardized_duration_estimands(
        dfit,
        target_data=raw_d.head(10),
        target_scale="raw",
        predictive_quantile=0.9,
        ndraws=20,
    )
    assert d_raw.metadata["target_rows"] == 10

    with pytest.raises(GP3BayesError):
        c.estimate_standardized_duration_estimands(
            SimpleNamespace(family="binary", fit_performed=True)
        )
    with pytest.raises(GP3BayesError):
        c.estimate_standardized_duration_estimands(
            dfit,
            predictive_quantile=1,
        )

    b_summary = c.summarise_estimand_draws(b_est)
    assert "probability_difference" in set(b_summary["quantity"])
    one = c.summarise_estimand_draws(
        b_est,
        "probability_difference",
        probs=(0.1, 0.5, 0.9),
    )
    assert len(one) == 1

    with pytest.raises(GP3BayesError):
        c.summarise_estimand_draws(b_est, "missing")
    with pytest.raises(GP3BayesError):
        c.summarise_estimand_draws(b_est, probs=(0.5, 0.1, 0.9))
    with pytest.raises(GP3BayesError):
        c.summarise_estimand_draws([1.0])
    with pytest.raises(GP3BayesError):
        c.summarise_estimand_draws([1.0, np.inf])

    b_alt_draws = b_est.draws.copy()
    b_alt_draws["probability_difference"] += 0.01
    b_alt = c.Estimand(
        "binary",
        b_est.primary_quantity,
        b_alt_draws,
        b_est.metadata,
    )
    sensitivity = c.compare_estimand_sensitivity(
        b_est,
        {"alt": b_alt},
    )
    assert sensitivity.status == "review"
    invariant = c.audit_estimand_invariance(
        b_est,
        b_alt,
        tolerance=0.02,
    )
    assert invariant.invariance_established

    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(object(), {"alt": b_alt})  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(b_est, {})
    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(b_est, {"bad": object()})  # type: ignore[dict-item]

    converted_draws = d_est.draws.copy()
    multiplier = 0.001
    for name in (
        "reference_average_conditional_median",
        "focal_average_conditional_median",
        "conditional_median_difference",
        "reference_predictive_quantile",
        "focal_predictive_quantile",
        "predictive_quantile_difference",
    ):
        converted_draws[name] *= multiplier
    converted = c.Estimand(
        "duration",
        d_est.primary_quantity,
        converted_draws,
        d_est.metadata,
    )
    unit_audit = c.audit_duration_unit_invariance(
        d_est,
        converted,
        multiplier,
        tolerance=1e-8,
    )
    assert unit_audit.invariance_established
    with pytest.raises(GP3BayesError):
        c.audit_duration_unit_invariance(b_est, converted, multiplier)


def test_design_support_missingness_fixed_random_and_separation(monkeypatch):
    sim, contract, prepared, spec, _ = _binary_objects(2110)
    data = sim.data.copy()

    assert ds._status(["pass", "review"]) == "review"
    assert ds._status(["pass", "fail"]) == "fail"
    assert ds._status(["pass"]) == "pass"

    with pytest.raises(GP3BayesError):
        ds._input(data)
    assert ds._input(data, contract)[1] is contract
    assert ds._input(prepared)[0].shape[0] == len(prepared.data)
    assert ds._input(spec)[2] is spec
    with pytest.raises(GP3BayesError):
        ds._input(object())

    clean = ds.audit_missingness_structure(data, contract)
    assert clean.status == "pass"

    missing = data.copy()
    missing.loc[missing.index[:10], "trial_covariate"] = np.nan
    review = ds.audit_missingness_structure(
        missing,
        contract,
        review_fraction=0.05,
        fail_fraction=0.5,
    )
    assert review.status in {"review", "fail"}
    assert not review.grouping_table.empty

    absent = ds.audit_missingness_structure(
        data.drop(columns="trial_covariate"),
        contract,
    )
    assert absent.status == "fail"
    assert "trial_covariate" in absent.absent_columns

    with pytest.raises(GP3BayesError):
        ds.audit_missingness_structure(
            data,
            contract,
            review_fraction=0.5,
            fail_fraction=0.2,
        )

    fixed = ds.audit_fixed_effect_design(data, contract)
    assert fixed.n_rows > 0
    assert len(fixed.singular_values) >= 1

    with pytest.raises(GP3BayesError):
        ds.audit_fixed_effect_design(
            data,
            contract,
            condition_number_review=0.5,
        )
    with pytest.raises(GP3BayesError):
        ds.audit_fixed_effect_design(
            data,
            contract,
            leverage_multiplier=0.5,
        )

    all_missing = data.copy()
    all_missing["selected"] = np.nan
    empty = ds.audit_fixed_effect_design(all_missing, contract)
    assert empty.status == "fail"
    assert empty.error is not None

    original_matrix = ds._fixed_model_matrix

    def explode(*args, **kwargs):
        raise RuntimeError("matrix failure")

    monkeypatch.setattr(ds, "_fixed_model_matrix", explode)
    failed = ds.audit_fixed_effect_design(data, contract)
    assert failed.status == "fail"
    assert "matrix failure" in failed.error
    monkeypatch.setattr(ds, "_fixed_model_matrix", original_matrix)

    monkeypatch.setattr(
        ds,
        "_fixed_model_matrix",
        lambda d, c: (np.zeros((len(d), 1)), ("x",)),
    )
    rank0 = ds.audit_fixed_effect_design(data, contract)
    assert rank0.rank == 0
    assert rank0.status == "fail"
    monkeypatch.setattr(ds, "_fixed_model_matrix", original_matrix)

    missing_participant = ds.audit_random_effects_support(
        data.drop(columns="participant_id"),
        contract,
    )
    assert missing_participant.status == "fail"
    with pytest.raises(GP3BayesError):
        ds.audit_random_effects_support(
            data,
            contract,
            minimum_repeated_rows=0,
        )

    slope_contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=True,
    )
    slope = ds.audit_random_effects_support(
        data,
        slope_contract,
        minimum_condition_cell_rows=1,
    )
    assert "random_slope_support" in set(slope.component_table["component"])

    design = ds.audit_design_support(
        data,
        contract,
        separation=True,
        strict_readiness=False,
    )
    assert "separation" in set(design.component_table["component"])
    assert design.strict_readiness is None
    assert ds.preflight_model_specification(spec).family == "binary"


def test_unified_validation_dispatch_and_stage_map(monkeypatch):
    _, contract, prepared, spec, _ = _binary_objects(2120)

    assert u._family(SimpleNamespace(family="binary")) == "binary"

    class PupilThing:
        pass

    assert u._family(PupilThing()) == "pupil"
    assert u._family(object()) is None
    assert u._recognized(contract)
    assert not u._recognized(object())

    contract_validation = u.validate_gp3bayes_object(contract)
    prepared_validation = u.validate_gp3bayes_object(prepared)
    spec_validation = u.validate_gp3bayes_object(spec)
    assert contract_validation.status == "pass"
    assert prepared_validation.status == "pass"
    assert spec_validation.status == "pass"

    bad = u.validate_gp3bayes_object(object())
    assert bad.status == "fail"
    with pytest.raises(GP3BayesError):
        u.validate_gp3bayes_object(object(), strict=True)
    with pytest.raises(GP3BayesError):
        u.validate_gp3bayes_object(contract, recursive=1)  # type: ignore[arg-type]

    binary_fit = SimpleNamespace(family="binary", fit_performed=True)
    duration_fit = SimpleNamespace(family="duration", fit_performed=True)
    pupil_fit = SimpleNamespace(family="pupil", fit_performed=True)

    monkeypatch.setattr(binary, "diagnose_binary_fit", lambda fit, **kw: "binary-d")
    monkeypatch.setattr(duration, "diagnose_duration_fit", lambda fit, **kw: "duration-d")
    monkeypatch.setattr(pupil, "diagnose_pupil_fit", lambda fit, **kw: "pupil-d")
    assert u.diagnose_model_fit(binary_fit) == "binary-d"
    assert u.diagnose_model_fit(duration_fit) == "duration-d"
    assert u.diagnose_model_fit(pupil_fit) == "pupil-d"

    monkeypatch.setattr(binary, "summarise_binary_posterior", lambda fit, **kw: "binary-s")
    monkeypatch.setattr(duration, "summarise_duration_posterior", lambda fit, **kw: "duration-s")
    monkeypatch.setattr(pupil, "summarise_pupil_posterior", lambda fit, **kw: "pupil-s")
    assert u.summarise_model_posterior(binary_fit) == "binary-s"
    assert u.summarise_model_posterior(duration_fit) == "duration-s"
    assert u.summarise_model_posterior(pupil_fit) == "pupil-s"

    monkeypatch.setattr(
        binary,
        "check_binary_posterior_predictive",
        lambda fit, **kw: "binary-p",
    )
    monkeypatch.setattr(
        duration,
        "check_duration_posterior_predictive",
        lambda fit, **kw: "duration-p",
    )
    monkeypatch.setattr(
        pupil,
        "check_pupil_posterior_predictive",
        lambda fit, **kw: "pupil-p",
    )
    assert u.check_model_ppc(binary_fit, draws=50) == "binary-p"
    assert u.check_model_ppc(duration_fit, draws=50) == "duration-p"
    assert u.check_model_ppc(pupil_fit, draws=50) == "pupil-p"

    monkeypatch.setattr(
        c,
        "estimate_standardized_probability_contrast",
        lambda fit: "binary-e",
    )
    monkeypatch.setattr(
        c,
        "estimate_standardized_duration_estimands",
        lambda fit: "duration-e",
    )
    assert u.estimate_model_estimands(binary_fit) == "binary-e"
    assert u.estimate_model_estimands(duration_fit) == "duration-e"
    with pytest.raises(GP3BayesError):
        u.estimate_model_estimands(pupil_fit)

    workflow = SimpleNamespace(
        fit=binary_fit,
        specification=spec,
        components={
            "diagnostics": object(),
            "posterior": object(),
            "ppc": object(),
            "estimands": object(),
            "sensitivity": object(),
            "loo": object(),
            "manifest": object(),
        },
    )
    stages = u.model_workflow_status(workflow)
    completed = dict(zip(stages["stage"], stages["completed"], strict=True))
    assert all(completed.values())
    assert stages.attrs["structural_stage_map_only"] is True
