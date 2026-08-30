from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

b = importlib.import_module("gp3bayespy.binary")
d = importlib.import_module("gp3bayespy.duration")


def _binary_spec(seed=5001):
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
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return gp.specify_binary_model(prepared)


def _duration_spec(seed=5002):
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
    prepared = gp.prepare_hierarchical_duration_data(sim.data, contract)
    return gp.specify_duration_model(prepared, baseline=500.0)


def test_contract_repr_dict_and_missing_attribute_branches():
    basic = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
    )
    text = repr(basic)
    assert "Item:" not in text
    assert "Condition:" not in text
    assert basic.as_dict()["interaction"] is None
    with pytest.raises(AttributeError):
        _ = basic.this_does_not_exist

    full = gp.create_model_contract(
        family="duration",
        outcome_col="y",
        participant_col="p",
        item_col="i",
        condition_col="c",
        predictors=("x", "z"),
        interaction=("x", "z"),
        outcome_unit="ms",
    )
    text = repr(full)
    assert "Item:" in text and "Condition:" in text and "Outcome unit:" in text
    assert full.as_dict()["interaction"] == ["x", "z"]


def test_binary_formula_matrix_summary_and_validation_guards():
    basic = gp.create_model_contract(family="binary", outcome_col="y", participant_col="p")
    assert b._fixed_formula_text(basic) == "y ~ 1"
    matrix, names = b._fixed_model_matrix(pd.DataFrame({"y": [0, 1], "p": ["a", "b"]}), basic)
    assert matrix.shape == (2, 1)
    assert names == ("(Intercept)",)

    inter = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x", "z"),
        interaction=("x", "z"),
    )
    data = pd.DataFrame({"y": [0, 1, 0], "p": ["a", "b", "c"], "x": [1, 2, 3], "z": [2, 3, 4]})
    formula = b._fixed_formula_text(inter)
    assert "x:z" in formula
    matrix, names = b._fixed_model_matrix(data, inter)
    assert any("x:z" in name for name in names)

    summary = b._binary_summary(
        np.array([0, 1, 1]),
        np.array([0.2, 0.8, 0.7]),
        None,
        np.array(["p1", "p1", "p1"]),
        None,
        (0.1, 0.9),
    )
    assert np.isnan(summary["condition_rate_contrast"])
    assert np.isnan(summary["item_rate_sd"])
    assert np.isnan(summary["participant_rate_sd"])

    one_level = b._binary_summary(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        np.array([0, 0]),
        np.array(["p1", "p2"]),
        np.array(["i1", "i1"]),
        (0.1, 0.9),
    )
    assert np.isnan(one_level["condition_rate_contrast"])
    assert np.isnan(one_level["item_rate_sd"])

    spec = _binary_spec()
    with pytest.raises(GP3BayesError):
        b._validate_binary_model_specification(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b._validate_binary_model_specification(replace(spec, family="duration"))
    with pytest.raises(GP3BayesError):
        b._validate_binary_model_specification(replace(spec, prepared=None))
    bad_audit = replace(spec.audit, ready=False)
    with pytest.raises(GP3BayesError):
        b._validate_binary_model_specification(replace(spec, audit=bad_audit))

    bad_link_contract = replace(
        spec.contract,
        template={**dict(spec.contract.template), "link": "probit"},
    )
    with pytest.raises(GP3BayesError):
        b._validate_binary_model_specification(replace(spec, contract=bad_link_contract))

    bad_like_contract = replace(
        spec.contract,
        template={**dict(spec.contract.template), "likelihood": "binomial"},
    )
    with pytest.raises(GP3BayesError):
        b._validate_binary_model_specification(replace(spec, contract=bad_like_contract))

    broken = replace(spec, prepared=None)
    with pytest.raises(GP3BayesError):
        b.check_binary_prior_predictive(broken)


def test_duration_contract_formula_summary_and_validation_guards():
    spec = _duration_spec()

    with pytest.raises(GP3BayesError):
        d._validate_duration_contract(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d._validate_duration_contract(replace(spec.contract, family="binary"))
    with pytest.raises(GP3BayesError):
        d._validate_duration_contract(
            replace(
                spec.contract,
                template={**dict(spec.contract.template), "likelihood": "gamma"},
            )
        )
    with pytest.raises(GP3BayesError):
        d._validate_duration_contract(replace(spec.contract, outcome_unit=""))

    basic = gp.create_model_contract(
        family="duration",
        outcome_col="y",
        participant_col="p",
        outcome_unit="ms",
    )
    assert d._fixed_formula_text(basic) == "y ~ 1"
    assert d._required_columns(basic) == ("y", "p")

    inter = gp.create_model_contract(
        family="duration",
        outcome_col="y",
        participant_col="p",
        predictors=("x", "z"),
        interaction=("x", "z"),
        outcome_unit="ms",
    )
    data = pd.DataFrame({"y": [1, 2, 3], "p": ["a", "b", "c"], "x": [1, 2, 3], "z": [2, 3, 4]})
    assert "x:z" in d._fixed_formula_text(inter)
    _, names = d._fixed_model_matrix(data, inter)
    assert any("x:z" in name for name in names)

    invalid = d._duration_summary(
        np.array([np.nan, -1.0]),
        None,
        np.array(["p1", "p1"]),
        None,
    )
    assert invalid["median"] == np.inf

    one = d._duration_summary(
        np.array([1.0]),
        np.array([0]),
        np.array(["p1"]),
        np.array(["i1"]),
    )
    assert np.isnan(one["coefficient_of_variation"])
    assert np.isnan(one["condition_median_ratio"])
    assert np.isnan(one["participant_log_median_sd"])
    assert np.isnan(one["item_log_median_sd"])

    assert d._positive_pair(None, 10.0) == (1.0, 100.0)

    with pytest.raises(GP3BayesError):
        d.specify_duration_model(object(), baseline=1.0)  # type: ignore[arg-type]
    bad_audit = replace(spec.audit, ready=False)
    bad_prepared = replace(spec.prepared, audit=bad_audit)
    with pytest.raises(GP3BayesError):
        d.specify_duration_model(bad_prepared, baseline=500.0)

    broken = replace(spec, prepared=None)
    with pytest.raises(GP3BayesError):
        d.check_duration_prior_predictive(broken)
