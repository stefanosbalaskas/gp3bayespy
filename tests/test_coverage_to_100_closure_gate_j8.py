from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

c = importlib.import_module("gp3bayespy.specification_closure")


def _binary(seed=5601):
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
        predictors=("trial_covariate",),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
        scale_predictors=("trial_covariate",),
    )
    return prepared, gp.specify_binary_model(prepared)


def _duration(seed=5602, condition=True):
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
        condition_col="condition" if condition else None,
        predictors=("trial_covariate",),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
        scale_predictors=("trial_covariate",),
    )
    return prepared, gp.specify_duration_model(prepared, baseline=500.0)


def test_duration_boundary_missing_outcome():
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="p",
        outcome_unit="milliseconds",
    )
    with pytest.raises(GP3BayesError, match="outcome column is absent"):
        c.audit_duration_boundaries(
            pd.DataFrame({"p": ["p1"]}),
            contract,
        )


def test_binary_recipe_outcome_absent_and_numeric_fallback():
    prepared, _ = _binary()
    recipe = c.create_transformation_recipe(prepared)
    raw = c.invert_transformation_recipe(prepared.data, recipe)

    no_outcome = raw.drop(columns=["selected"])
    replay = c.apply_transformation_recipe(
        no_outcome,
        recipe,
        require_outcome=False,
    )
    assert "selected" not in replay

    modified = copy.deepcopy(recipe)
    modified.transformations["outcome"]["mapping"] = {
        "no": 0,
        "yes": 1,
    }
    numeric = raw.copy()
    numeric["selected"] = np.resize([0, 1], len(numeric))
    replay2 = c.apply_transformation_recipe(numeric, modified)
    assert set(replay2["selected"]) <= {0, 1}


def test_duration_recipe_outcome_absent_numeric_condition_and_no_condition():
    prepared, _ = _duration()
    recipe = c.create_transformation_recipe(prepared)
    raw = c.invert_transformation_recipe(prepared.data, recipe)

    no_outcome = raw.drop(columns=["duration"])
    replay = c.apply_transformation_recipe(
        no_outcome,
        recipe,
        require_outcome=False,
    )
    assert "duration" not in replay

    encoded = raw.copy()
    condition = prepared.contract.mappings["condition"]
    coding = recipe.transformations["condition"]["coding"]
    encoded[condition] = np.resize(
        list(coding.values()),
        len(encoded),
    )
    encoded_replay = c.apply_transformation_recipe(encoded, recipe)
    assert np.isfinite(encoded_replay[condition]).all()

    prepared2, _ = _duration(5603, condition=False)
    recipe2 = c.create_transformation_recipe(prepared2)
    raw2 = c.invert_transformation_recipe(prepared2.data, recipe2)
    replay2 = c.apply_transformation_recipe(raw2, recipe2)
    assert len(replay2) == len(raw2)


def test_random_slope_plan_exception_branch(monkeypatch):
    _, spec = _binary(5610)
    monkeypatch.setattr(
        c,
        "_reprepare",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("synthetic")),
    )
    plan = c.create_random_slope_sensitivity_plan(spec)
    assert not plan.intercept_only["ready"]
    assert not plan.random_slope["ready"]
    assert "synthetic" in plan.intercept_only["error"]


def test_fit_spec_binary_and_duration_dispatch(monkeypatch):
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
    binary_token = object()
    duration_token = object()
    monkeypatch.setattr(
        aow,
        "fit_binary_model_backend",
        lambda *a, **k: binary_token,
    )
    monkeypatch.setattr(
        aow,
        "fit_duration_model_backend",
        lambda *a, **k: duration_token,
    )

    args = dict(
        backend="analytic",
        chains=1,
        iter=20,
        warmup=10,
        cores=1,
        seed=1,
        adapt_delta=0.9,
        max_treedepth=10,
        refresh=0,
    )
    assert c._fit_spec(SimpleNamespace(family="binary"), **args) is binary_token
    assert c._fit_spec(SimpleNamespace(family="duration"), **args) is duration_token
