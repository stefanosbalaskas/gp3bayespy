from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed: int = 3001):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
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
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("trial_covariate",),
    )
    return sim.data.copy(), prepared


def _duration(seed: int = 3011):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
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
    return sim.data.copy(), prepared


def test_binary_transformation_recipe_guard_and_inverse_matrix():
    raw, prepared = _binary()
    recipe = c.create_transformation_recipe(prepared)
    assert recipe.family == "binary"
    assert c._as_recipe(prepared).family == "binary"

    with pytest.raises(GP3BayesError):
        c.create_transformation_recipe(object())
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe([], recipe)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            raw,
            recipe,
            input_scale="bad",
        )

    no_outcome = raw.drop(columns="selected")
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            no_outcome,
            recipe,
            require_outcome=True,
        )

    prepared_scale = c.apply_transformation_recipe(
        prepared.data,
        recipe,
        input_scale="prepared",
    )
    assert prepared_scale.attrs["gp3bayes_transformation_recipe"] is recipe

    bad_outcome = raw.copy()
    bad_outcome["selected"] = bad_outcome["selected"].astype(object)
    bad_outcome.loc[bad_outcome.index[0], "selected"] = "unknown"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_outcome, recipe)

    bad_condition = raw.copy()
    bad_condition["condition"] = bad_condition["condition"].astype(object)
    bad_condition.loc[bad_condition.index[0], "condition"] = "unknown"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_condition, recipe)

    missing_scaled = raw.drop(columns="trial_covariate")
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(missing_scaled, recipe)

    bad_scaled = raw.copy()
    bad_scaled.loc[bad_scaled.index[0], "trial_covariate"] = np.inf
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_scaled, recipe)

    applied = c.apply_transformation_recipe(raw, recipe)
    assert applied.attrs["gp3bayes_transformation_recipe"] is recipe

    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe([], recipe)  # type: ignore[arg-type]

    bad_prepared_condition = prepared.data.copy()
    bad_prepared_condition.loc[bad_prepared_condition.index[0], "condition"] = 99.0
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(
            bad_prepared_condition,
            recipe,
        )

    bad_prepared_outcome = prepared.data.copy()
    bad_prepared_outcome.loc[bad_prepared_outcome.index[0], "selected"] = 9
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(
            bad_prepared_outcome,
            recipe,
        )

    audit = c.validate_transformation_replay(prepared)
    assert audit.replay_established
    assert audit.status == "pass"
    with pytest.raises(GP3BayesError):
        c.validate_transformation_replay(prepared, tolerance=-1)


def test_duration_transformation_recipe_guard_and_inverse_matrix():
    raw, prepared = _duration()
    recipe = c.create_transformation_recipe(prepared)
    assert recipe.family == "duration"

    source_unit = recipe.transformations["outcome"]["source_unit"]
    wrong_unit = "seconds" if source_unit != "seconds" else "milliseconds"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            raw,
            recipe,
            input_unit=wrong_unit,
        )

    bad_duration = raw.copy()
    bad_duration.loc[bad_duration.index[0], "duration"] = 0.0
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            bad_duration,
            recipe,
            input_unit=source_unit,
        )

    missing_condition = raw.drop(columns="condition")
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            missing_condition,
            recipe,
            input_unit=source_unit,
        )

    bad_condition = raw.copy()
    bad_condition["condition"] = bad_condition["condition"].astype(object)
    bad_condition.loc[bad_condition.index[0], "condition"] = "unknown"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            bad_condition,
            recipe,
            input_unit=source_unit,
        )

    missing_scaled = raw.drop(columns="trial_covariate")
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            missing_scaled,
            recipe,
            input_unit=source_unit,
        )

    bad_scaled = raw.copy()
    bad_scaled.loc[bad_scaled.index[0], "trial_covariate"] = np.inf
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            bad_scaled,
            recipe,
            input_unit=source_unit,
        )

    applied = c.apply_transformation_recipe(
        raw,
        recipe,
        input_unit=source_unit,
    )
    assert np.isfinite(pd.to_numeric(applied["duration"], errors="coerce")).all()

    bad_prepared = prepared.data.copy()
    bad_prepared.loc[bad_prepared.index[0], "condition"] = 99.0
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(bad_prepared, recipe)

    restored = c.invert_transformation_recipe(prepared.data, recipe)
    assert len(restored) == len(prepared.data)

    audit = c.validate_transformation_replay(prepared)
    assert audit.replay_established
