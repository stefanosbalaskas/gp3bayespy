from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.readiness as r
from gp3bayespy.exceptions import GP3BayesError


def _collector():
    rows = []

    def add(check_id, category, status, message, n_affected=None):
        rows.append((check_id, category, status, message, n_affected))

    return rows, add


def test_readiness_scalar_identifier_and_validation_helpers():
    for value in (None, pd.NA, "x", b"x", True, 1, 1.2, np.int64(2)):
        assert r._supported_scalar(value)
    assert not r._supported_scalar(1 + 2j)
    assert not r._supported_scalar({"x": 1})

    assert r._supported_identifier(pd.Series(["a", "b"]))
    assert r._supported_identifier(pd.Series([1, 2]))
    assert r._supported_identifier(pd.Series([True, False]))
    assert not r._supported_identifier(pd.Series(pd.to_datetime(["2020-01-01"])))
    assert not r._supported_identifier(pd.Series(pd.to_timedelta(["1 day"])))
    assert not r._supported_identifier(pd.Series([1 + 1j]))
    assert not r._supported_identifier(pd.Series([{"x": 1}], dtype=object))

    assert r._supported_predictor(pd.Series(["a"]))
    assert r._is_categorical(pd.Series(["a", "b"]))
    assert r._is_categorical(pd.Series(pd.Categorical(["a", "b"])))
    assert not r._is_categorical(pd.Series([1.0, 2.0]))
    assert r._n_unique(pd.Series([1, 1, 2, np.nan])) == 2

    data = pd.DataFrame({"x": [1, 2]})
    assert r._observed_levels(data, "x") == 2
    assert r._observed_levels(data, None) is None
    assert r._observed_levels(data, "missing") is None

    with pytest.raises(GP3BayesError):
        r._validate_data([1, 2])  # type: ignore[arg-type]

    bad_names = pd.DataFrame([[1, 2]])
    bad_names.columns = ["", "x"]
    with pytest.raises(GP3BayesError):
        r._validate_data(bad_names)

    dup = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(GP3BayesError):
        r._validate_data(dup)

    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
    )
    r._validate_contract(contract)
    with pytest.raises(GP3BayesError):
        r._validate_contract(object())  # type: ignore[arg-type]


def test_identifier_and_outcome_audit_branch_matrix():
    rows, add = _collector()
    r._audit_identifier_types(
        pd.DataFrame({"x": [1]}),
        {"participant": "p", "item": None, "trial": None},
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_identifier_types(
        pd.DataFrame({"p": pd.to_datetime(["2020-01-01"])}),
        {"participant": "p", "item": None, "trial": None},
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_identifier_types(
        pd.DataFrame({"p": ["p1", "p2"]}),
        {"participant": "p", "item": None, "trial": None},
        add,
    )
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_binary_outcome(pd.Series(["no", "yes"]), add)
    assert rows[-1][0] == "outcome_type"
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_binary_outcome(pd.Series([0.0, 2.0]), add)
    statuses = {x[0]: x[2] for x in rows}
    assert statuses["outcome_values"] == "fail"
    assert statuses["outcome_support"] == "fail"

    rows, add = _collector()
    r._audit_binary_outcome(pd.Series([0, 1, 0, 1]), add)
    assert all(x[2] == "pass" for x in rows)

    rows, add = _collector()
    r._audit_duration_outcome(pd.Series(["1", "2"]), add)
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_duration_outcome(pd.Series([1.0, 0.0, np.inf]), add)
    statuses = {x[0]: x[2] for x in rows}
    assert statuses["duration_finite"] == "fail"
    assert statuses["duration_positive"] == "fail"

    rows, add = _collector()
    r._audit_duration_outcome(pd.Series([1.0, 1.0]), add)
    assert {x[0]: x[2] for x in rows}["outcome_support"] == "fail"

    rows, add = _collector()
    r._audit_duration_outcome(pd.Series([1.0, 2.0, 3.0]), add)
    assert all(x[2] == "pass" for x in rows)


def test_participant_item_trial_structure_branch_matrix():
    rows, add = _collector()
    r._audit_participant_structure(pd.DataFrame({"p": ["p1"]}), "p", add)
    assert {x[0]: x[2] for x in rows}["participant_levels"] == "fail"

    rows, add = _collector()
    r._audit_participant_structure(pd.DataFrame({"p": ["p1", "p2"]}), "p", add)
    assert {x[0]: x[2] for x in rows}["repeated_measurement"] == "fail"

    rows, add = _collector()
    r._audit_participant_structure(pd.DataFrame({"p": ["p1", "p1", "p2"]}), "p", add)
    assert {x[0]: x[2] for x in rows}["repeated_measurement"] == "warn"

    rows, add = _collector()
    r._audit_participant_structure(pd.DataFrame({"p": ["p1", "p1", "p2", "p2"]}), "p", add)
    assert all(x[2] == "pass" for x in rows)

    rows, add = _collector()
    r._audit_item_structure(pd.DataFrame({"p": ["p1"]}), "p", None, add)
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_item_structure(
        pd.DataFrame({"p": ["p1", "p2"], "item": ["i1", "i1"]}),
        "p",
        "item",
        add,
    )
    assert {x[0]: x[2] for x in rows}["item_levels"] == "fail"

    rows, add = _collector()
    r._audit_item_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p1", "p2", "p2"],
                "item": ["i1", "i2", "i1", "i2"],
            }
        ),
        "p",
        "item",
        add,
    )
    assert {x[0]: x[2] for x in rows}["item_crossing"] == "pass"

    rows, add = _collector()
    r._audit_item_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p2"],
                "item": ["i1", "i2"],
            }
        ),
        "p",
        "item",
        add,
    )
    assert {x[0]: x[2] for x in rows}["item_crossing"] == "warn"

    rows, add = _collector()
    r._audit_trial_structure(
        pd.DataFrame({"p": ["p1", "p1"], "trial": [1, 1]}),
        "p",
        "trial",
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_trial_structure(
        pd.DataFrame({"p": ["p1", "p1"], "trial": [1, 2]}),
        "p",
        "trial",
        add,
    )
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_trial_structure(pd.DataFrame({"p": ["p1"]}), "p", None, add)
    assert rows[-1][2] == "pass"


def test_condition_time_and_predictor_branch_matrix():
    rows, add = _collector()
    r._audit_condition_structure(
        pd.DataFrame({"p": ["p1"]}),
        "p",
        None,
        False,
        add,
    )
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_condition_structure(
        pd.DataFrame({"p": ["p1", "p2"], "c": pd.to_datetime(["2020-01-01", "2020-01-02"])}),
        "p",
        "c",
        False,
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_condition_structure(
        pd.DataFrame({"p": ["p1", "p1"], "c": ["A", "A"]}),
        "p",
        "c",
        True,
        add,
    )
    assert {x[0]: x[2] for x in rows}["condition_levels"] == "fail"

    rows, add = _collector()
    r._audit_condition_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p1", "p2", "p2"],
                "c": ["A", "B", "A", "A"],
            }
        ),
        "p",
        "c",
        True,
        add,
    )
    statuses = {x[0]: x[2] for x in rows}
    assert statuses["random_slope_support"] == "fail"
    assert statuses["random_slope_replication"] == "warn"

    rows, add = _collector()
    r._audit_condition_structure(
        pd.DataFrame(
            {
                "p": ["p1"] * 4 + ["p2"] * 4,
                "c": ["A", "A", "B", "B"] * 2,
            }
        ),
        "p",
        "c",
        True,
        add,
    )
    statuses = {x[0]: x[2] for x in rows}
    assert statuses["random_slope_support"] == "pass"
    assert statuses["random_slope_replication"] == "pass"

    rows, add = _collector()
    r._audit_time_structure(pd.DataFrame({"p": ["p1"]}), "p", None, add)
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_time_structure(
        pd.DataFrame({"p": ["p1", "p1"], "t": ["a", "b"]}),
        "p",
        "t",
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_time_structure(
        pd.DataFrame({"p": ["p1", "p1"], "t": [1.0, np.inf]}),
        "p",
        "t",
        add,
    )
    assert {x[0]: x[2] for x in rows}["time_finite"] == "fail"

    rows, add = _collector()
    r._audit_time_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p1", "p2", "p2"],
                "t": [1.0, 1.0, 1.0, 2.0],
            }
        ),
        "p",
        "t",
        add,
    )
    assert {x[0]: x[2] for x in rows}["time_within_participant"] == "warn"

    rows, add = _collector()
    r._audit_predictor_structure(pd.DataFrame({"x": [1]}), [], add)
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_predictor_structure(
        pd.DataFrame({"x": pd.to_datetime(["2020-01-01", "2020-01-02"])}),
        ["x"],
        add,
    )
    assert {x[0]: x[2] for x in rows}["predictor_types"] == "fail"

    rows, add = _collector()
    r._audit_predictor_structure(
        pd.DataFrame({"x": [1.0, np.inf]}),
        ["x"],
        add,
    )
    assert {x[0]: x[2] for x in rows}["predictor_finite"] == "fail"

    rows, add = _collector()
    r._audit_predictor_structure(
        pd.DataFrame({"x": [1.0, 1.0]}),
        ["x"],
        add,
    )
    assert {x[0]: x[2] for x in rows}["predictor_variation"] == "fail"

    rows, add = _collector()
    r._audit_predictor_structure(
        pd.DataFrame({"x": ["a", " "]}),
        ["x"],
        add,
    )
    assert {x[0]: x[2] for x in rows}["predictor_blanks"] == "fail"

    rows, add = _collector()
    cat = pd.Categorical(["a", "b"], categories=["a", "b", "unused"])
    r._audit_predictor_structure(pd.DataFrame({"x": cat}), ["x"], add)
    assert {x[0]: x[2] for x in rows}["predictor_factor_levels"] == "warn"


def _interaction_contract():
    return gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("a", "b"),
        interaction=("a", "b"),
    )


def test_interaction_audit_all_major_paths():
    contract = _interaction_contract()

    rows, add = _collector()
    r._audit_interaction_structure(
        pd.DataFrame({"p": ["p1"], "y": [0], "a": ["A"]}),
        contract,
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_interaction_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p2"],
                "y": [0, 1],
                "a": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "b": ["x", "y"],
            }
        ),
        contract,
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_interaction_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p2"],
                "y": [0, 1],
                "a": ["A", "A"],
                "b": ["x", "y"],
            }
        ),
        contract,
        add,
    )
    assert rows[-1][2] == "fail"

    rows, add = _collector()
    r._audit_interaction_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p2", "p3"],
                "y": [0, 1, 0],
                "a": [1.0, 2.0, 3.0],
                "b": ["x", "x", "y"],
            }
        ),
        contract,
        add,
    )
    assert rows[-1][2] == "pass"

    rows, add = _collector()
    r._audit_interaction_structure(
        pd.DataFrame(
            {
                "p": ["p1", "p2", "p3"],
                "y": [0, 1, 0],
                "a": ["A", "A", "B"],
                "b": ["x", "y", "x"],
            }
        ),
        contract,
        add,
    )
    assert rows[-1][2] == "warn"

    rows, add = _collector()
    r._audit_interaction_structure(
        pd.DataFrame(
            {
                "p": ["p1"] * 4,
                "y": [0, 1, 0, 1],
                "a": ["A", "A", "B", "B"],
                "b": ["x", "x", "y", "y"],
            }
        ),
        contract,
        add,
    )
    assert rows[-1][2] == "pass"

    no_interaction = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
    )
    rows, add = _collector()
    r._audit_interaction_structure(pd.DataFrame({"y": [0], "p": ["p1"]}), no_interaction, add)
    assert rows[-1][2] == "pass"


def test_full_readiness_empty_failure_and_repr():
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        condition_col="c",
        predictors=("x",),
    )
    empty = pd.DataFrame(columns=["y", "p", "c", "x"])
    audit = r.audit_model_readiness(empty, contract)
    assert not audit.ready
    assert "Ready: FALSE" in repr(audit)

    data = pd.DataFrame(
        {
            "y": [0, 1, 0, 1],
            "p": ["p1", "p1", "p2", "p2"],
            "c": ["A", "B", "A", "B"],
            "x": [0.0, 1.0, 0.5, 1.5],
        }
    )
    audit2 = r.audit_model_readiness(data, contract)
    assert audit2.ready
    assert audit2.status in {"ready", "ready_with_warnings"}
