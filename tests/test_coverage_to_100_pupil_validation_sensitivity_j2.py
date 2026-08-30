from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _prepared(seed: int = 3801):
    sim = p.simulate_pupil_timecourse(
        n_participants=2,
        trials_per_participant=3,
        n_items=2,
        sampling_frequency=10,
        time_window=(-0.2, 0.5),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0,
        seed=seed,
    )
    contract = p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time",
        pupil_unit="millimetres",
        sampling_frequency=10,
        item_col="item_id",
        condition_col="condition",
        baseline_window=(-0.2, 0.0),
    )
    return p.prepare_pupil_timecourse(sim.data, contract)


def test_pupil_validation_plan_all_targets_and_guards():
    prepared = _prepared()

    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(object())

    tiny = replace(prepared, data=prepared.data.head(3).copy())
    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(tiny)

    plans = {}
    for target in (
        "new_sample_known_trial",
        "new_trial_known_participant",
        "new_participant",
        "future_segment",
    ):
        plan = p.create_pupil_validation_plan(
            prepared,
            target=target,
            K=3,
            future_fraction=0.25,
            seed=3802,
        )
        assert not plan.leakage_detected
        plans[target] = plan

    one_participant = replace(
        prepared,
        data=prepared.data[
            prepared.data[".participant"] == prepared.data[".participant"].iloc[0]
        ].copy(),
    )
    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(
            one_participant,
            target="new_participant",
        )

    one_trial_each = prepared.data.groupby(
        ".participant",
        observed=True,
        sort=False,
    ).head(8)
    one_trial_each = replace(prepared, data=one_trial_each.copy())
    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(
            one_trial_each,
            target="new_trial_known_participant",
        )

    short_series = prepared.data.groupby(".series_id", observed=True, sort=False).head(2).copy()
    short_prepared = replace(prepared, data=short_series)
    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(
            short_prepared,
            target="future_segment",
            future_fraction=0.5,
        )

    with pytest.raises(GP3BayesError):
        p.create_pupil_validation_plan(
            prepared,
            target="bad",  # type: ignore[arg-type]
        )

    spec = p.specify_pupil_timecourse_model(
        prepared,
        autocorrelation="none",
    )
    fit = p.fit_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=80,
        warmup=40,
        cores=1,
        seed=3803,
    )

    with pytest.raises(GP3BayesError):
        p.validate_pupil_model(object(), plans["new_participant"])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.validate_pupil_model(fit, object())  # type: ignore[arg-type]

    leaking = replace(
        plans["new_participant"],
        leakage_detected=True,
    )
    with pytest.raises(GP3BayesError):
        p.validate_pupil_model(fit, leaking)

    for plan in plans.values():
        validation = p.validate_pupil_model(
            fit,
            plan,
            execute=False,
        )
        assert not validation.executed
        assert not p.pupil_validation_table(validation).empty

    with pytest.raises(GP3BayesError):
        p.pupil_validation_table(object())


def test_pupil_sensitivity_suite_materialization_and_tables():
    prepared = _prepared(3810)
    spec = p.specify_pupil_timecourse_model(
        prepared,
        autocorrelation="none",
        smooth_basis_dimension=6,
    )

    with pytest.raises(GP3BayesError):
        p.create_pupil_sensitivity_suite(object())  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        p.create_pupil_sensitivity_suite(
            spec,
            baseline_windows=((-0.2, 0.0),),
        )

    suite = p.create_pupil_sensitivity_suite(
        spec,
        baseline_windows=((-0.2, 0.0),),
        baseline_window_operation="subtract",
        baseline_operations=("subtract", "divide"),
        interpolation_policy=("exclude",),
        blink_adjacent_margins=(0.05,),
        gaze_adjustment=("none",),
        luminance_adjustment=("none",),
        pfe_prepared={"alt": prepared},
        smooth_basis_dimensions=(5, 8),
        autocorrelation=("none", "ar1"),
        analysis_windows=((0.0, 0.2),),
    )
    table = p.pupil_sensitivity_table(suite)
    assert len(table) >= 10

    for axis in (
        "smooth_basis_dimension",
        "autocorrelation",
        "analysis_window",
        "pfe_prepared",
        "baseline_operation",
    ):
        scenario_id = table.loc[
            table["axis"] == axis,
            "scenario_id",
        ].iloc[0]
        materialized = p.materialize_pupil_sensitivity_scenario(
            suite,
            scenario_id,
        )
        assert materialized["scenario_id"] == scenario_id

    with pytest.raises(GP3BayesError):
        p.materialize_pupil_sensitivity_scenario(object(), "S001")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.materialize_pupil_sensitivity_scenario(suite, "missing")

    estimand_a = p.PupilEstimand(
        table=pd.DataFrame({"estimate": [1.0]}),
        estimand="window_mean",
        unit="millimetres",
        probability=0.95,
        window=(0.0, 0.2),
    )
    estimand_b = p.PupilEstimand(
        table=pd.DataFrame({"estimate": [2.0]}),
        estimand="window_mean",
        unit="millimetres",
        probability=0.95,
        window=(0.0, 0.2),
    )
    comparison = p.compare_pupil_sensitivity_estimands({"a": estimand_a, "b": estimand_b})
    assert set(comparison.table["scenario_id"]) == {"a", "b"}
    assert len(p.pupil_sensitivity_table(comparison)) == 2

    with pytest.raises(GP3BayesError):
        p.compare_pupil_sensitivity_estimands({})
    with pytest.raises(GP3BayesError):
        p.compare_pupil_sensitivity_estimands({"bad": object()})  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_sensitivity_table(object())
