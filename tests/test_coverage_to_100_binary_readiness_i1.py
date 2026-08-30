from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as b
import gp3bayespy.readiness as r
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed: int = 2701):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=5,
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
    return sim.data.copy(), contract


def test_binary_helper_and_preparation_guard_matrix():
    data, contract = _binary()

    with pytest.raises(GP3BayesError):
        b._numeric_scalar(True, "x")
    with pytest.raises(GP3BayesError):
        b._numeric_scalar(2, "x", upper=1)
    with pytest.raises(GP3BayesError):
        b._integer(1.5, "n")
    with pytest.raises(GP3BayesError):
        b._flag(1, "flag")
    assert b._character_vector(None, "x") == ()
    with pytest.raises(GP3BayesError):
        b._character_vector(("a", "a"), "x")
    with pytest.raises(GP3BayesError):
        b._validate_binary_contract(object())  # type: ignore[arg-type]

    duration_contract = replace(contract, family="duration")
    with pytest.raises(GP3BayesError):
        b._validate_binary_contract(duration_contract)

    with pytest.raises(GP3BayesError):
        b._quote_name("")
    assert b._quote_name("if").startswith("`")
    assert b._quote_name("a b").startswith("`")

    bool_outcome = pd.Series([True, False])
    mapped, mapping = b._map_binary_outcome(bool_outcome, None)
    assert mapped.tolist() == [1, 0]
    assert mapping == {0: 0, 1: 1}

    numeric = pd.Series([0, 1, 1])
    mapped, _ = b._map_binary_outcome(numeric, None)
    assert mapped.tolist() == [0, 1, 1]

    labels = pd.Series(["no", "yes"])
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, None)
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, {"no": 0, "yes": 0})
    with pytest.raises(GP3BayesError):
        b._map_binary_outcome(labels, {"no": 0, "other": 1})
    mapped, stored = b._map_binary_outcome(labels, {"no": 0, "yes": 1})
    assert mapped.tolist() == [0, 1]
    assert stored["yes"] == 1

    with pytest.raises(GP3BayesError):
        b._code_condition(pd.Series(["A", "B"]), None, (0.5, 0.5))
    with pytest.raises(GP3BayesError):
        b._code_condition(pd.Series(["A"]), None, (-0.5, 0.5))
    with pytest.raises(GP3BayesError):
        b._code_condition(
            pd.Series(["A", "B"]),
            ("A", "C"),
            (-0.5, 0.5),
        )

    cat = pd.Series(pd.Categorical(["B", "A", "B"], categories=["A", "B"]))
    coded, levels, coding = b._code_condition(cat, None, (-0.5, 0.5))
    assert levels == ("A", "B")
    assert set(coded.unique()) == {-0.5, 0.5}
    assert coding["A"] == -0.5

    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            [],  # type: ignore[arg-type]
            contract,
        )
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            data,
            contract,
            missing="bad",
        )
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            data,
            contract,
            scale_time=1,  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            data,
            contract,
            scale_predictors=("condition",),
        )
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            data.drop(columns="participant_id"),
            contract,
        )

    missing = data.copy()
    missing.loc[missing.index[0], "trial_covariate"] = np.nan
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            missing,
            contract,
            missing="error",
        )
    dropped = b.prepare_hierarchical_binary_data(
        missing,
        contract,
        missing="drop",
    )
    assert dropped.rows_removed == 1

    all_missing = data.copy()
    all_missing["trial_covariate"] = np.nan
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            all_missing,
            contract,
            missing="drop",
        )

    nonnumeric = data.copy()
    nonnumeric["trial_covariate"] = "x"
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            nonnumeric,
            contract,
            scale_predictors=("trial_covariate",),
        )

    constant = data.copy()
    constant["trial_covariate"] = 1.0
    with pytest.raises(GP3BayesError):
        b.prepare_hierarchical_binary_data(
            constant,
            contract,
            scale_predictors=("trial_covariate",),
        )

    prepared = b.prepare_hierarchical_binary_data(data, contract)
    with pytest.raises(GP3BayesError):
        b.specify_binary_model(object())  # type: ignore[arg-type]

    assert b._probability_pair((0.1, 0.9), "p") == (0.1, 0.9)
    with pytest.raises(GP3BayesError):
        b._probability_pair((0.9, 0.1), "p")

    spec = b.specify_binary_model(prepared)
    with pytest.raises(GP3BayesError):
        b.check_binary_prior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        b.check_binary_prior_predictive(spec, draws=49)


def test_readiness_helper_guard_matrix():
    data, contract = _binary(2710)

    assert r._supported_scalar(None)
    assert r._supported_scalar(pd.NA)
    assert r._supported_scalar("x")
    assert not r._supported_scalar(1 + 2j)
    assert not r._supported_scalar({"x": 1})

    assert not r._supported_identifier(pd.Series(pd.to_datetime(["2026-01-01"])))
    assert not r._supported_identifier(pd.Series(pd.to_timedelta(["1 day"])))
    assert not r._supported_identifier(pd.Series(np.array([1 + 1j], dtype=complex)))
    assert not r._supported_identifier(pd.Series([{"x": 1}], dtype=object))
    assert r._supported_predictor(pd.Series(["a", "b"]))
    assert r._is_categorical(pd.Series(["a", "b"]))
    assert not r._is_categorical(pd.Series([1.0, 2.0]))
    assert r._observed_levels(data, None) is None
    assert r._observed_levels(data, "missing") is None

    with pytest.raises(GP3BayesError):
        r._validate_data(object())  # type: ignore[arg-type]

    empty_name = pd.DataFrame([[1]], columns=[""])
    with pytest.raises(GP3BayesError):
        r._validate_data(empty_name)

    dup = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(GP3BayesError):
        r._validate_data(dup)

    with pytest.raises(GP3BayesError):
        r._validate_contract(object())  # type: ignore[arg-type]

    broken = replace(
        contract,
        mappings={key: value for key, value in contract.mappings.items() if key != "trial"},
    )
    with pytest.raises(GP3BayesError):
        r._validate_contract(broken)

    seen = []

    def add(check_id, category, status, message, n_affected=None):
        seen.append((check_id, category, status, message, n_affected))

    r._audit_identifier_types(
        data,
        {"participant": None, "item": None, "trial": None},
        add,
    )
    assert seen
    assert seen[0][0] == "identifier_types"
    assert seen[0][2] == "fail"
    assert seen[0][4] == 0

    no_participant = replace(
        contract,
        mappings={**contract.mappings, "participant": None},
    )
    with pytest.raises(
        GP3BayesError,
        match="non-null outcome and participant mappings",
    ):
        r.audit_model_readiness(data, no_participant)
