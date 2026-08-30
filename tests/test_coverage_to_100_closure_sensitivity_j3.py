from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed=4101, *, item=True):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=5,
        trials_per_participant=8,
        n_items=4,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id" if item else None,
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data if item else sim.data.drop(columns=["item_id"]),
        contract,
        scale_predictors=("trial_covariate",),
    )
    spec = gp.specify_binary_model(prepared)
    return spec


def _duration(seed=4102):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=5,
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
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("trial_covariate",),
    )
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    return spec


def _estimand(family="binary", shift=0.0):
    if family == "binary":
        draws = pd.DataFrame(
            {
                ".draw": [1, 2, 3, 4],
                "probability_difference": np.array([0.1, 0.2, 0.3, 0.4]) + shift,
            }
        )
        return c.Estimand(
            "binary",
            "probability_difference",
            draws,
            {},
        )
    base = np.array([1.0, 1.1, 1.2, 1.3])
    draws = pd.DataFrame(
        {
            ".draw": [1, 2, 3, 4],
            "conditional_median_ratio": base + shift,
            "predictive_quantile_ratio": base + shift,
            "reference_average_conditional_median": 500 * base,
            "focal_average_conditional_median": 550 * base,
            "reference_predictive_quantile": 800 * base,
            "focal_predictive_quantile": 900 * base,
        }
    )
    return c.Estimand(
        "duration",
        "conditional_median_ratio",
        draws,
        {},
    )


def test_estimand_metadata_target_summary_and_sensitivity_guards():
    bspec = _binary()

    assert c._condition_metadata(bspec.prepared)["column"] == "condition"

    no_condition = replace(
        bspec.prepared,
        transformations={
            **dict(bspec.prepared.transformations),
            "condition": None,
        },
    )
    with pytest.raises(GP3BayesError):
        c._condition_metadata(no_condition)

    bad_condition = replace(
        bspec.prepared,
        transformations={
            **dict(bspec.prepared.transformations),
            "condition": {
                "column": "condition",
                "source_levels": ("a", "b", "c"),
                "coding": {"a": -1, "b": 0, "c": 1},
            },
        },
    )
    with pytest.raises(GP3BayesError):
        c._condition_metadata(bad_condition)

    fake_fit = SimpleNamespace(specification=bspec)
    target = bspec.prepared.data.head(3).copy()
    assert len(c._target_data(fake_fit, None, "prepared")) == len(bspec.prepared.data)
    assert len(c._target_data(fake_fit, target, "prepared")) == 3
    with pytest.raises(GP3BayesError):
        c._target_data(fake_fit, target, "bad")

    ref = _estimand()
    alt = _estimand(shift=0.05)
    assert len(c.summarise_estimand_draws(ref)) == 1
    assert len(c.summarise_estimand_draws([1.0, 2.0, 3.0])) == 1
    with pytest.raises(GP3BayesError):
        c.summarise_estimand_draws(ref, quantities="missing")
    with pytest.raises(GP3BayesError):
        c.summarise_estimand_draws(ref, probs=(0.5, 0.4, 0.9))
    with pytest.raises(GP3BayesError):
        c._summary_vector([1.0], (0.025, 0.5, 0.975))
    with pytest.raises(GP3BayesError):
        c._summary_vector([1.0, np.nan], (0.025, 0.5, 0.975))

    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(object(), {"a": alt})  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(ref, {})
    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(ref, {"a": alt}, "missing")
    with pytest.raises(GP3BayesError):
        c.compare_estimand_sensitivity(ref, {"a": object()})  # type: ignore[arg-type]

    comparison = c.compare_estimand_sensitivity(ref, {"a": alt})
    assert comparison.status == "review"

    same = c.audit_estimand_invariance(ref, ref, tolerance=0)
    assert same.invariance_established
    changed = c.audit_estimand_invariance(ref, alt, tolerance=0)
    assert not changed.invariance_established

    with pytest.raises(GP3BayesError):
        c.audit_estimand_invariance(ref, alt, tolerance=-1)


def test_sensitivity_plan_builders_binary_duration_and_guards(monkeypatch):
    bspec = _binary(4110)

    with pytest.raises(GP3BayesError):
        c.create_random_slope_sensitivity_plan(object())

    plan = c.create_random_slope_sensitivity_plan(bspec)
    assert plan.intercept_only["contract"].random_slope is False
    assert plan.random_slope["contract"].random_slope is True

    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(bspec, group="bad")
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(bspec, max_units=1)

    units = tuple(bspec.prepared.data["participant_id"].astype(str).unique()[:2])
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(bspec, units=(units[0], units[0]))
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(bspec, units=("unknown",))

    deletion = c.create_group_deletion_sensitivity_plan(
        bspec,
        units=units,
    )
    assert len(deletion.table) == 2

    no_item = _binary(4111, item=False)
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(
            no_item,
            group="item",
        )

    original_audit = c.audit_model_readiness
    monkeypatch.setattr(
        c,
        "audit_model_readiness",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic")),
    )
    errored = c.create_group_deletion_sensitivity_plan(
        bspec,
        units=(units[0],),
    )
    assert errored.table.iloc[0]["status"] == "error"
    monkeypatch.setattr(c, "audit_model_readiness", original_audit)

    with pytest.raises(GP3BayesError):
        c.create_contrast_coding_sensitivity_specification(bspec, (0, 0), 0.5)
    recoded = c.create_contrast_coding_sensitivity_specification(bspec, (-1, 1), 0.5)
    assert recoded.prepared.transformations["condition"]["coding"]

    with pytest.raises(GP3BayesError):
        c.create_predictor_scaling_sensitivity_specification(bspec, "missing", 2, 0.5)
    with pytest.raises(GP3BayesError):
        c.create_predictor_scaling_sensitivity_specification(bspec, "participant_covariate", 2, 0.5)

    scaled = c.create_predictor_scaling_sensitivity_specification(bspec, "trial_covariate", 2, 0.5)
    assert scaled.family == "binary"

    dspec = _duration(4112)
    dscaled = c.create_predictor_scaling_sensitivity_specification(dspec, "trial_covariate", 2, 0.5)
    assert dscaled.family == "duration"

    with pytest.raises(GP3BayesError):
        c.create_duration_unit_sensitivity_specification(bspec, 0.001, "seconds")
    with pytest.raises(GP3BayesError):
        c.create_duration_unit_sensitivity_specification(dspec, 0.001, "")
    converted = c.create_duration_unit_sensitivity_specification(dspec, 0.001, "seconds")
    assert converted.outcome_unit == "seconds"


def test_duration_unit_invariance_and_sensitivity_run_orchestration(monkeypatch):
    ref = _estimand("duration")
    converted_draws = ref.draws.copy()
    for name in (
        "reference_average_conditional_median",
        "focal_average_conditional_median",
        "reference_predictive_quantile",
        "focal_predictive_quantile",
    ):
        converted_draws[name] = converted_draws[name] * 0.001
    converted = c.Estimand(
        "duration",
        "conditional_median_ratio",
        converted_draws,
        {},
    )

    audit = c.audit_duration_unit_invariance(ref, converted, multiplier=0.001, tolerance=1e-12)
    assert audit.invariance_established

    changed = c.audit_duration_unit_invariance(ref, ref, multiplier=0.001, tolerance=0)
    assert not changed.invariance_established

    with pytest.raises(GP3BayesError):
        c.audit_duration_unit_invariance(_estimand(), converted, 0.001)

    bspec = _binary(4120)
    deletion = c.create_group_deletion_sensitivity_plan(
        bspec,
        units=tuple(bspec.prepared.data["participant_id"].astype(str).unique()[:2]),
    )

    fake_fit = SimpleNamespace(family="binary")
    fake_est = _estimand("binary")

    monkeypatch.setattr(c, "_fit_spec", lambda *args, **kwargs: fake_fit)
    monkeypatch.setattr(c, "_primary_estimand", lambda *args, **kwargs: fake_est)

    run = c.run_group_deletion_sensitivity(
        deletion,
        backend="analytic",
        chains=1,
        iter=20,
        warmup=10,
        cores=1,
        retain_fits=True,
    )
    assert set(run.summary["status"]) == {"completed"}
    assert run.reference_fit is fake_fit
    assert run.fits is not None

    run2 = c.run_group_deletion_sensitivity(
        deletion,
        backend="analytic",
        chains=1,
        iter=20,
        warmup=10,
        cores=1,
        retain_fits=False,
    )
    assert run2.reference_fit is None
    assert run2.fits is None

    with pytest.raises(GP3BayesError):
        c.run_group_deletion_sensitivity(object())  # type: ignore[arg-type]

    ready_plan = c.RandomSlopeSensitivityPlan(
        "0.2",
        "binary",
        {"ready": True, "specification": bspec},
        {"ready": True, "specification": bspec},
    )
    result = c.run_random_slope_sensitivity(
        ready_plan,
        backend="analytic",
        chains=1,
        iter=20,
        warmup=10,
        cores=1,
        retain_fits=True,
    )
    assert result["fits"] is not None
    assert result["automatic_selection"] is False

    with pytest.raises(GP3BayesError):
        c.run_random_slope_sensitivity(object())  # type: ignore[arg-type]

    blocked = c.RandomSlopeSensitivityPlan(
        "0.2",
        "binary",
        {"ready": False, "specification": None},
        {"ready": True, "specification": bspec},
    )
    with pytest.raises(GP3BayesError):
        c.run_random_slope_sensitivity(blocked)
