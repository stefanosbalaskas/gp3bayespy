from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as b
import gp3bayespy.duration as d
from gp3bayespy.exceptions import GP3BayesError


def _bcontract(*, condition=True, item=True):
    return gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id" if item else None,
        trial_col="trial_id",
        condition_col="condition" if condition else None,
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=False,
    )


def _dcontract(*, condition=True, item=True):
    return gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id" if item else None,
        trial_col="trial_id",
        condition_col="condition" if condition else None,
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=False,
        outcome_unit="milliseconds",
    )


def test_binary_foundation_helpers_and_simulation_edges():
    assert b._quote_name("alpha") == "alpha"
    assert b._quote_name("a.b_1") == "a.b_1"
    assert b._quote_name("if") == "`if`"
    assert b._quote_name("1 bad").startswith("`")
    assert "\\`" in b._quote_name("a`b")
    with pytest.raises(GP3BayesError):
        b._quote_name("")

    with pytest.raises(GP3BayesError):
        b._character_vector(["x", "x"], "x")
    with pytest.raises(GP3BayesError):
        b._character_vector([""], "x")
    assert b._character_vector(None, "x") == ()
    assert b._character_vector("x", "x") == ("x",)

    assert b._column_basis(pd.Series([True, False]), "z")[0][0] == "z"
    assert b._column_basis(pd.Series([1.0, 2.0]), "z")[0][0] == "z"
    one = pd.Series(pd.Categorical(["a", "a"], categories=["a"]))
    assert b._column_basis(one, "z") == []
    cat = pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
    assert len(b._column_basis(cat, "z")) == 1

    mapped, mapping = b._map_binary_outcome(pd.Series([True, False]), None)
    assert set(mapped) == {0, 1}
    assert mapping == {0: 0, 1: 1}

    mapped, mapping = b._map_binary_outcome(pd.Series([0.0, 1.0]), None)
    assert set(mapped) == {0, 1}

    labels = pd.Series(["no", "yes", "no"])
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, None)
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, {"no": 0})
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, {"no": 0, "yes": 2})
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, {"no": 0, "other": 1})
    mapped, mapping = b._map_binary_outcome(labels, {"no": 0, "yes": 1})
    assert mapped.tolist() == [0, 1, 0]
    assert mapping == {"no": 0, "yes": 1}

    cond = pd.Series(pd.Categorical(["b", "a", "b", "a"], categories=["a", "b"]))
    coded, levels, coding = b._code_condition(cond, None, (-0.5, 0.5))
    assert levels == ("a", "b")
    assert set(coded) == {-0.5, 0.5}
    assert coding["a"] == -0.5

    numeric = pd.Series([2, 1, 2, 1])
    _, levels, _ = b._code_condition(numeric, None, (-1, 1))
    assert levels == ("1", "2")

    with pytest.raises(GP3BayesError):
        b._code_condition(cond, None, (0, 0))
    with pytest.raises(GP3BayesError):
        b._code_condition(pd.Series(["a", "a"]), None, (-1, 1))
    with pytest.raises(GP3BayesError):
        b._code_condition(cond, ("a", "c"), (-1, 1))

    sim = b.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=5,
        n_items=3,
        balanced_condition=False,
        include_items=False,
        condition_probability=0.4,
        seed=4001,
    )
    assert "item_id" not in sim.data
    assert sim.random_effects["item"] is None
    assert sim.design["include_items"] is False
    assert "Rows:" in repr(sim)

    for kwargs in (
        {"random_slope_cor": 1.0},
        {"condition_probability": 0.0},
        {"balanced_condition": 1},
        {"include_items": 1},
    ):
        with pytest.raises(GP3BayesError):
            b.simulate_hierarchical_binary_data(
                n_participants=3,
                trials_per_participant=4,
                n_items=2,
                **kwargs,
            )


def test_binary_preparation_guard_and_no_condition_ppc(monkeypatch):
    sim = b.simulate_hierarchical_binary_data(
        n_participants=5,
        trials_per_participant=6,
        n_items=3,
        seed=4010,
    )
    contract = _bcontract()

    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data([], contract)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(sim.data, _dcontract())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(sim.data, contract, missing="bad")
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(sim.data, contract, scale_predictors=("missing",))
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(sim.data.drop(columns="trial_covariate"), contract)

    missing = sim.data.copy()
    missing.loc[missing.index[0], "trial_covariate"] = np.nan
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(missing, contract)
    dropped = b.prepare_hierarchical_binary_data(missing, contract, missing="drop")
    assert dropped.rows_removed == 1

    all_missing = sim.data.copy()
    all_missing["trial_covariate"] = np.nan
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(all_missing, contract, missing="drop")

    labelled = sim.data.copy()
    labelled["selected"] = labelled["selected"].map({0: "no", 1: "yes"})
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(labelled, contract)
    labelled_prepared = b.prepare_hierarchical_binary_data(
        labelled,
        contract,
        outcome_mapping={"no": 0, "yes": 1},
    )
    assert set(labelled_prepared.data["selected"]) == {0, 1}

    text_scale = sim.data.copy()
    text_scale["trial_covariate"] = "x"
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            text_scale,
            contract,
            scale_predictors=("trial_covariate",),
        )

    inf_scale = sim.data.copy()
    inf_scale.loc[inf_scale.index[0], "trial_covariate"] = np.inf
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            inf_scale,
            contract,
            scale_predictors=("trial_covariate",),
        )

    constant = sim.data.copy()
    constant["trial_covariate"] = 1.0
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            constant,
            contract,
            scale_predictors=("trial_covariate",),
        )

    original = b._fixed_model_matrix
    monkeypatch.setattr(
        b,
        "_fixed_model_matrix",
        lambda data, contract: (
            np.ones((len(data), 2)),
            ("(Intercept)", "duplicate"),
        ),
    )
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(sim.data, contract)
    monkeypatch.setattr(b, "_fixed_model_matrix", original)

    no_cond_contract = _bcontract(condition=False, item=False)
    no_cond = b.prepare_hierarchical_binary_data(
        sim.data.drop(columns=["item_id"]),
        no_cond_contract,
        scale_predictors=("trial_covariate",),
    )
    assert no_cond.transformations["condition"] is None
    assert "not_applicable" in set(no_cond.decision_log["value"])

    spec = b.specify_binary_model(no_cond)
    ppc = b.check_binary_prior_predictive(
        spec,
        draws=50,
        seed=4011,
        maximum_extreme_probability=1.0,
    )
    assert "not_applicable" in set(ppc.checks["status"])
    assert "Draws:" in repr(ppc)

    with pytest.raises(GP3BayesError):
        b._probability_pair((0.5, 0.5), "x")
    with pytest.raises(GP3BayesError):
        b.check_binary_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.check_binary_prior_predictive(spec, draws=10)


def test_duration_foundation_preparation_and_ppc_edges(monkeypatch):
    sim = d.simulate_hierarchical_duration_data(
        n_participants=5,
        trials_per_participant=6,
        n_items=3,
        balanced_condition=False,
        include_items=False,
        condition_probability=0.4,
        outcome_unit="ms",
        seed=4020,
    )
    assert "item_id" not in sim.data
    assert sim.random_effects["item"] is None
    assert sim.design["include_items"] is False
    assert "Outcome unit:" in repr(sim)

    for kwargs in (
        {"random_slope_cor": -1.0},
        {"condition_probability": 1.0},
        {"outcome_unit": ""},
    ):
        with pytest.raises(GP3BayesError):
            d.simulate_hierarchical_duration_data(
                n_participants=3,
                trials_per_participant=4,
                n_items=2,
                **kwargs,
            )

    contract = _dcontract(item=False)
    data = sim.data.copy()

    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data([], contract)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(data, _bcontract(item=False))  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(data, contract, missing="bad")
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(data, contract, scale_predictors=("missing",))
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(
            data,
            contract,
            outcome_multiplier=0.001,
        )
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(
            data,
            contract,
            converted_unit="",
        )

    converted = d.prepare_hierarchical_duration_data(
        data,
        contract,
        outcome_multiplier=0.001,
        converted_unit="seconds",
        scale_predictors=("trial_covariate",),
    )
    assert converted.outcome_unit == "seconds"
    assert converted.transformations["outcome"]["multiplier"] == 0.001

    missing = data.copy()
    missing.loc[missing.index[0], "trial_covariate"] = np.nan
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(missing, contract)
    dropped = d.prepare_hierarchical_duration_data(missing, contract, missing="drop")
    assert dropped.rows_removed == 1

    invalid_y = data.copy()
    invalid_y.loc[invalid_y.index[0], "duration"] = np.inf
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(invalid_y, contract)
    invalid_y.loc[invalid_y.index[0], "duration"] = 0.0
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(invalid_y, contract)

    text_scale = data.copy()
    text_scale["trial_covariate"] = "x"
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(
            text_scale,
            contract,
            scale_predictors=("trial_covariate",),
        )
    constant = data.copy()
    constant["trial_covariate"] = 1.0
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(
            constant,
            contract,
            scale_predictors=("trial_covariate",),
        )

    original = d._fixed_model_matrix
    monkeypatch.setattr(
        d,
        "_fixed_model_matrix",
        lambda data, contract: (
            np.ones((len(data), 2)),
            ("(Intercept)", "duplicate"),
        ),
    )
    with pytest.raises(GP3BayesError):
        d.prepare_hierarchical_duration_data(data, contract)
    monkeypatch.setattr(d, "_fixed_model_matrix", original)

    no_cond_contract = _dcontract(condition=False, item=False)
    no_cond = d.prepare_hierarchical_duration_data(
        data,
        no_cond_contract,
        scale_predictors=("trial_covariate",),
    )
    assert no_cond.transformations["condition"] is None

    spec = d.specify_duration_model(no_cond, baseline=500.0)
    ppc = d.check_duration_prior_predictive(
        spec,
        draws=50,
        seed=4021,
        maximum_extreme_probability=1.0,
    )
    assert "not_applicable" in set(ppc.checks["status"])
    assert "Outcome unit:" in repr(ppc)

    with pytest.raises(GP3BayesError):
        d._positive_pair((1.0, 1.0), 500.0)
    with pytest.raises(GP3BayesError):
        d.check_duration_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        d.check_duration_prior_predictive(spec, draws=10)
