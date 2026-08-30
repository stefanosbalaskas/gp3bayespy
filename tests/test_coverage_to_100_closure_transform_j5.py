from __future__ import annotations

import numpy as np
import pytest

import gp3bayespy as gp
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed=4601):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed
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
    return gp.prepare_hierarchical_binary_data(
        sim.data, contract, scale_predictors=("trial_covariate",)
    )


def _duration(seed=4602):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed
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
    return gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("trial_covariate",),
        outcome_multiplier=0.001,
        converted_unit="seconds",
    )


def test_transformation_recipe_binary_full_guard_matrix():
    prepared = _binary()
    recipe = c.create_transformation_recipe(prepared)
    assert c._as_recipe(recipe) is recipe
    with pytest.raises(GP3BayesError):
        c.create_transformation_recipe(object())

    raw = c.invert_transformation_recipe(prepared.data, recipe)
    replay = c.apply_transformation_recipe(raw, recipe)
    assert replay.attrs["gp3bayes_transformation_recipe"] is recipe

    prepared_input = c.apply_transformation_recipe(prepared.data, recipe, input_scale="prepared")
    assert prepared_input.attrs["gp3bayes_transformation_recipe"] is recipe

    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe([], recipe)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(raw, recipe, input_scale="bad")
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(
            raw.drop(columns=["selected"]),
            recipe,
            require_outcome=True,
        )

    bad_outcome = raw.copy()
    bad_outcome["selected"] = "unknown"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_outcome, recipe)

    condition_col = recipe.transformations["condition"]["column"]
    bad_condition = raw.copy()
    bad_condition[condition_col] = "unknown"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_condition, recipe)

    missing_condition = raw.drop(columns=[condition_col])
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(missing_condition, recipe)

    missing_scaled = raw.drop(columns=["trial_covariate"])
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(missing_scaled, recipe)

    bad_scaled = raw.copy()
    bad_scaled.loc[bad_scaled.index[0], "trial_covariate"] = np.inf
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_scaled, recipe)

    encoded = raw.copy()
    coding = recipe.transformations["condition"]["coding"]
    encoded[condition_col] = encoded[condition_col].map(coding)
    c.apply_transformation_recipe(encoded, recipe)

    invalid_prepared_condition = prepared.data.copy()
    invalid_prepared_condition[condition_col] = 999
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(invalid_prepared_condition, recipe)

    invalid_prepared_outcome = prepared.data.copy()
    invalid_prepared_outcome["selected"] = 9
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(invalid_prepared_outcome, recipe)

    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe([], recipe)  # type: ignore[arg-type]

    audit = c.validate_transformation_replay(prepared)
    assert audit.replay_established
    with pytest.raises(GP3BayesError):
        c.validate_transformation_replay(prepared, tolerance=-1)


def test_transformation_recipe_duration_full_guard_matrix():
    prepared = _duration()
    recipe = c.create_transformation_recipe(prepared)
    raw = c.invert_transformation_recipe(prepared.data, recipe)
    replay = c.apply_transformation_recipe(raw, recipe, input_unit="milliseconds")
    assert len(replay) == len(prepared.data)

    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(raw, recipe, input_unit="seconds")

    bad_y = raw.copy()
    bad_y.loc[bad_y.index[0], "duration"] = 0.0
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_y, recipe)

    bad_y.loc[bad_y.index[0], "duration"] = np.inf
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_y, recipe)

    condition_col = prepared.contract.mappings["condition"]
    assert condition_col is not None
    bad_condition = raw.copy()
    bad_condition[condition_col] = "unknown"
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_condition, recipe)

    missing_condition = raw.drop(columns=[condition_col])
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(missing_condition, recipe)

    missing_scaled = raw.drop(columns=["trial_covariate"])
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(missing_scaled, recipe)

    bad_scaled = raw.copy()
    bad_scaled.loc[bad_scaled.index[0], "trial_covariate"] = np.nan
    with pytest.raises(GP3BayesError):
        c.apply_transformation_recipe(bad_scaled, recipe)

    invalid_prepared_condition = prepared.data.copy()
    invalid_prepared_condition[condition_col] = 999
    with pytest.raises(GP3BayesError):
        c.invert_transformation_recipe(invalid_prepared_condition, recipe)

    restored = c.invert_transformation_recipe(prepared.data, recipe)
    assert np.all(restored["duration"] > 0)


def test_transformation_recipe_without_condition_paths():
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=3, trials_per_participant=5, n_items=2, seed=4610
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col=None,
        predictors=("trial_covariate",),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data, contract, scale_predictors=("trial_covariate",)
    )
    recipe = c.create_transformation_recipe(prepared)
    raw = c.invert_transformation_recipe(prepared.data, recipe)
    replay = c.apply_transformation_recipe(raw, recipe)
    assert len(replay) == len(raw)
