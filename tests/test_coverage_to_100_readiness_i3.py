from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

import gp3bayespy as gp
import gp3bayespy.readiness as r


def _collector():
    rows = []

    def add(check_id, category, status, message, n_affected=None):
        rows.append((check_id, category, status, message, n_affected))

    return rows, add


def _contract(random_slope: bool = False):
    return gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="participant",
        item_col="item",
        trial_col="trial",
        condition_col="condition",
        predictors=("x",),
        random_slope=random_slope,
    )


def test_readiness_internal_branch_matrix_participant_item_trial():
    rows, add = _collector()

    r._audit_participant_structure(pd.DataFrame({"x": [1]}), "participant", add)
    assert rows == []

    rows.clear()
    r._audit_participant_structure(
        pd.DataFrame({"participant": [{"x": 1}, {"x": 2}]}),
        "participant",
        add,
    )
    assert rows == []

    rows.clear()
    r._audit_participant_structure(
        pd.DataFrame({"participant": ["p1", "p1"]}),
        "participant",
        add,
    )
    assert {x[0] for x in rows} == {
        "participant_levels",
        "repeated_measurement",
    }
    assert rows[0][2] == "fail"

    rows.clear()
    r._audit_participant_structure(
        pd.DataFrame({"participant": ["p1", "p2"]}),
        "participant",
        add,
    )
    assert rows[-1][2] == "fail"

    rows.clear()
    r._audit_participant_structure(
        pd.DataFrame({"participant": ["p1", "p1", "p2"]}),
        "participant",
        add,
    )
    assert rows[-1][2] == "warn"

    rows.clear()
    r._audit_item_structure(
        pd.DataFrame({"participant": ["p1"]}),
        "participant",
        None,
        add,
    )
    assert rows[0][0] == "item_structure"

    rows.clear()
    r._audit_item_structure(
        pd.DataFrame({"participant": ["p1"]}),
        "participant",
        "item",
        add,
    )
    assert rows == []

    rows.clear()
    r._audit_item_structure(
        pd.DataFrame(
            {
                "participant": [{"a": 1}, {"a": 2}],
                "item": ["i1", "i2"],
            }
        ),
        "participant",
        "item",
        add,
    )
    assert rows[-1][0] == "item_crossing"
    assert rows[-1][2] == "fail"

    rows.clear()
    r._audit_item_structure(
        pd.DataFrame(
            {
                "participant": ["p1", "p2"],
                "item": ["i1", "i2"],
            }
        ),
        "participant",
        "item",
        add,
    )
    assert rows[-1][2] == "warn"

    rows.clear()
    r._audit_trial_structure(
        pd.DataFrame({"participant": ["p1"]}),
        "participant",
        None,
        add,
    )
    assert rows[0][0] == "trial_key"

    rows.clear()
    r._audit_trial_structure(
        pd.DataFrame(
            {
                "participant": ["p1", "p1"],
                "trial": [1, 1],
            }
        ),
        "participant",
        "trial",
        add,
    )
    assert rows[0][2] == "fail"

    rows.clear()
    r._audit_trial_structure(
        pd.DataFrame(
            {
                "participant": ["p1", "p1"],
                "trial": [1, 2],
            }
        ),
        "participant",
        "trial",
        add,
    )
    assert rows[0][2] == "pass"


def test_readiness_condition_time_predictor_interaction_branch_matrix():
    rows, add = _collector()

    base = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2"],
            "condition": ["A", "B", "A", "B"],
            "time": [0.0, 1.0, 0.0, 1.0],
        }
    )

    r._audit_condition_structure(base, "participant", None, False, add)
    assert rows[-1][0] == "condition_levels"

    rows.clear()
    unsupported = base.copy()
    unsupported["condition"] = [{"a": 1}] * len(unsupported)
    r._audit_condition_structure(unsupported, "participant", "condition", False, add)
    assert rows[-1][0] == "condition_type"
    assert rows[-1][2] == "fail"

    rows.clear()
    one = base.copy()
    one["condition"] = "A"
    r._audit_condition_structure(one, "participant", "condition", False, add)
    assert any(x[0] == "condition_levels" and x[2] == "fail" for x in rows)
    assert rows[-1][0] == "random_slope_support"

    rows.clear()
    bad_participant = base.copy()
    bad_participant["participant"] = [{"a": 1}] * len(base)
    r._audit_condition_structure(
        bad_participant,
        "participant",
        "condition",
        True,
        add,
    )
    assert rows[-1][2] == "fail"

    rows.clear()
    insufficient = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2"],
            "condition": ["A", "A", "A", "B"],
        }
    )
    r._audit_condition_structure(insufficient, "participant", "condition", True, add)
    assert any(x[0] == "random_slope_support" and x[2] == "fail" for x in rows)
    assert any(x[0] == "random_slope_replication" and x[2] == "warn" for x in rows)

    rows.clear()
    r._audit_time_structure(base, "participant", None, add)
    assert rows[-1][0] == "time_structure"

    rows.clear()
    bad_time = base.copy()
    bad_time["time"] = ["a", "b", "c", "d"]
    r._audit_time_structure(bad_time, "participant", "time", add)
    assert rows[-1][0] == "time_type"
    assert rows[-1][2] == "fail"

    rows.clear()
    inf_time = base.copy()
    inf_time["time"] = [0.0, np.inf, 0.0, np.inf]
    r._audit_time_structure(inf_time, "participant", "time", add)
    assert any(x[0] == "time_finite" and x[2] == "fail" for x in rows)
    assert any(x[0] == "time_variation" and x[2] == "fail" for x in rows)

    rows.clear()
    no_within = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2"],
            "time": [0.0, 0.0, 1.0, 1.0],
        }
    )
    r._audit_time_structure(no_within, "participant", "time", add)
    assert rows[-1][2] == "fail"

    rows.clear()
    mixed = pd.DataFrame(
        {
            "participant": ["p1", "p1", "p2", "p2"],
            "time": [0.0, 1.0, 1.0, 1.0],
        }
    )
    r._audit_time_structure(mixed, "participant", "time", add)
    assert rows[-1][2] == "warn"

    rows.clear()
    r._audit_predictor_structure(base, [], add)
    assert rows[-1][0] == "predictor_structure"

    rows.clear()
    r._audit_predictor_structure(base, ["missing"], add)
    assert rows == []

    rows.clear()
    pred = pd.DataFrame(
        {
            "bad": [{"a": 1}, {"a": 2}],
            "num": [1.0, np.inf],
            "constant": [1.0, 1.0],
            "text": ["", "x"],
            "cat": pd.Categorical(
                ["a", "a"],
                categories=["a", "unused"],
            ),
        }
    )
    r._audit_predictor_structure(
        pred,
        ["bad", "num", "constant", "text", "cat"],
        add,
    )
    statuses = {(x[0], x[2]) for x in rows}
    assert ("predictor_types", "fail") in statuses
    assert ("predictor_finite", "fail") in statuses
    assert ("predictor_variation", "fail") in statuses
    assert ("predictor_blanks", "fail") in statuses
    assert ("predictor_factor_levels", "warn") in statuses

    contract = _contract()
    rows.clear()
    r._audit_interaction_structure(base, contract, add)
    assert rows[-1][0] == "interaction_support"

    rows.clear()
    missing_contract = replace(
        contract,
        interaction=("condition", "missing"),
    )
    r._audit_interaction_structure(base, missing_contract, add)
    assert rows[-1][2] == "fail"

    rows.clear()
    unsupported_data = base.copy()
    unsupported_data["bad"] = [{"a": 1}] * len(base)
    unsupported_contract = replace(
        contract,
        interaction=("condition", "bad"),
    )
    r._audit_interaction_structure(
        unsupported_data,
        unsupported_contract,
        add,
    )
    assert rows[-1][2] == "fail"

    rows.clear()
    invariant_data = base.copy()
    invariant_data["constant"] = "x"
    invariant_contract = replace(
        contract,
        interaction=("condition", "constant"),
    )
    r._audit_interaction_structure(
        invariant_data,
        invariant_contract,
        add,
    )
    assert rows[-1][2] == "fail"

    rows.clear()
    numeric_data = base.copy()
    numeric_data["numeric"] = [0.0, 1.0, 2.0, 3.0]
    numeric_contract = replace(
        contract,
        interaction=("condition", "numeric"),
    )
    r._audit_interaction_structure(
        numeric_data,
        numeric_contract,
        add,
    )
    assert rows[-1][2] == "pass"

    rows.clear()
    sparse = pd.DataFrame(
        {
            "participant": ["p1", "p2", "p3"],
            "condition": ["A", "A", "B"],
            "group": ["X", "Y", "Y"],
        }
    )
    sparse_contract = replace(
        contract,
        interaction=("condition", "group"),
    )
    r._audit_interaction_structure(sparse, sparse_contract, add)
    assert rows[-1][2] in {"warn", "pass"}
