from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.advanced_optional_workflows as aow
import gp3bayespy.readiness as r
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _collector():
    rows = []

    def add(check_id, category, status, message, n_affected=None):
        rows.append((check_id, category, status, message, n_affected))

    return rows, add


def _binary(seed=4301, *, random_slope=False):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=8,
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
        random_slope=random_slope,
    )
    return sim.data.copy(), contract


def _duration(seed=4302):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=8,
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
    return sim.data.copy(), contract


def test_readiness_scalar_dtype_and_validation_guards():
    assert r._supported_scalar(None)
    assert r._supported_scalar(pd.NA)
    assert r._supported_scalar(np.int64(1))
    assert not r._supported_scalar(1 + 2j)
    assert not r._supported_scalar({"x": 1})

    assert not r._supported_identifier(pd.Series(pd.to_datetime(["2026-01-01"])))
    assert not r._supported_identifier(pd.Series(pd.to_timedelta(["1 day"])))
    assert not r._supported_identifier(pd.Series([1 + 2j]))
    assert r._supported_identifier(pd.Series(["a", "b"], dtype=object))
    assert not r._supported_identifier(pd.Series([{"a": 1}, {"b": 2}], dtype=object))
    assert r._is_categorical(pd.Series(["a", "b"]))
    assert not r._is_categorical(pd.Series([1.0, 2.0]))

    with pytest.raises(GP3BayesError):
        r._validate_data([])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        r._validate_data(pd.DataFrame([[1]], columns=[""]))
    with pytest.raises(GP3BayesError):
        r._validate_data(pd.DataFrame([[1, 2]], columns=["x", "x"]))

    _, contract = _binary()
    with pytest.raises(GP3BayesError):
        r._validate_contract(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        r._validate_contract(replace(contract, mappings={"outcome": "selected"}))

    assert r._observed_levels(pd.DataFrame({"x": [1]}), None) is None
    assert r._observed_levels(pd.DataFrame({"x": [1]}), "missing") is None
    assert r._observed_levels(pd.DataFrame({"x": [1, 1, 2]}), "x") == 2


def test_readiness_structure_helper_branch_matrix():
    rows, add = _collector()
    data, _ = _binary(4310)

    r._audit_item_structure(data, "participant_id", None, add)
    assert rows[-1][0] == "item_structure"

    rows.clear()
    one_item = data.copy()
    one_item["item_id"] = "i1"
    r._audit_item_structure(one_item, "participant_id", "item_id", add)
    assert any(x[0] == "item_levels" and x[2] == "fail" for x in rows)

    rows.clear()
    bad_item = data.copy()
    bad_item["item_id"] = pd.to_datetime("2026-01-01")
    r._audit_item_structure(bad_item, "participant_id", "item_id", add)
    assert rows == []

    rows.clear()
    weak = pd.DataFrame(
        {
            "participant_id": ["p1", "p2"],
            "item_id": ["i1", "i2"],
        }
    )
    r._audit_item_structure(weak, "participant_id", "item_id", add)
    assert any(x[0] == "item_crossing" and x[2] == "warn" for x in rows)

    rows.clear()
    r._audit_trial_structure(data, "participant_id", None, add)
    assert rows[-1][0] == "trial_key"

    rows.clear()
    dup = data.head(2).copy()
    dup["participant_id"] = "p1"
    dup["trial_id"] = "t1"
    r._audit_trial_structure(dup, "participant_id", "trial_id", add)
    assert rows[-1][2] == "fail"

    rows.clear()
    r._audit_condition_structure(data, "participant_id", None, False, add)
    assert rows[-1][0] == "condition_levels"

    rows.clear()
    bad_condition = data.copy()
    bad_condition["condition"] = pd.to_datetime("2026-01-01")
    r._audit_condition_structure(
        bad_condition,
        "participant_id",
        "condition",
        False,
        add,
    )
    assert rows[-1][0] == "condition_type"
    assert rows[-1][2] == "fail"

    rows.clear()
    one_condition = data.copy()
    one_condition["condition"] = 0
    r._audit_condition_structure(
        one_condition,
        "participant_id",
        "condition",
        False,
        add,
    )
    assert any(x[0] == "condition_levels" and x[2] == "fail" for x in rows)

    rows.clear()
    slope = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "condition": [0, 0, 1, 1],
        }
    )
    r._audit_condition_structure(
        slope,
        "participant_id",
        "condition",
        True,
        add,
    )
    assert any(x[0] == "random_slope_support" and x[2] == "fail" for x in rows)

    rows.clear()
    weak_cells = pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2"],
            "condition": [0, 1, 0, 1],
        }
    )
    r._audit_condition_structure(
        weak_cells,
        "participant_id",
        "condition",
        True,
        add,
    )
    assert any(x[0] == "random_slope_replication" and x[2] == "warn" for x in rows)


def test_readiness_time_helper_branches():
    rows, add = _collector()

    r._audit_time_structure(pd.DataFrame({"p": ["a", "a"]}), "p", None, add)
    assert rows[-1][0] == "time_structure"

    rows.clear()
    bad = pd.DataFrame({"p": ["a", "a"], "t": ["x", "y"]})
    r._audit_time_structure(bad, "p", "t", add)
    assert rows[-1][0] == "time_type"
    assert rows[-1][2] == "fail"

    rows.clear()
    inf = pd.DataFrame({"p": ["a", "a"], "t": [0.0, np.inf]})
    r._audit_time_structure(inf, "p", "t", add)
    assert any(x[0] == "time_finite" and x[2] == "fail" for x in rows)

    rows.clear()
    constant = pd.DataFrame({"p": ["a", "a", "b", "b"], "t": [1.0, 1.0, 1.0, 1.0]})
    r._audit_time_structure(constant, "p", "t", add)
    assert any(x[0] == "time_variation" and x[2] == "fail" for x in rows)
    assert any(x[0] == "time_within_participant" and x[2] == "fail" for x in rows)

    rows.clear()
    partial = pd.DataFrame(
        {
            "p": ["a", "a", "b", "b"],
            "t": [1.0, 2.0, 1.0, 1.0],
        }
    )
    r._audit_time_structure(partial, "p", "t", add)
    assert any(x[0] == "time_within_participant" and x[2] == "warn" for x in rows)

    rows.clear()
    good = pd.DataFrame(
        {
            "p": ["a", "a", "b", "b"],
            "t": [1.0, 2.0, 3.0, 4.0],
        }
    )
    r._audit_time_structure(good, "p", "t", add)
    assert any(x[0] == "time_within_participant" and x[2] == "pass" for x in rows)


def test_strict_readiness_rank_and_separation_branches(monkeypatch):
    data, contract = _binary(4320)

    original_matrix = c._closure_fixed_model_matrix
    monkeypatch.setattr(
        c,
        "_closure_fixed_model_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic-rank")),
    )
    audit = c.audit_model_readiness_strict(
        data,
        contract,
        run_separation=False,
    )
    assert audit.rank["rank"] is None
    assert audit.status == "not_ready"
    monkeypatch.setattr(c, "_closure_fixed_model_matrix", original_matrix)

    monkeypatch.setattr(
        aow,
        "detect_binary_separation",
        lambda *args, **kwargs: SimpleNamespace(separation_detected=True),
    )
    detected = c.audit_model_readiness_strict(
        data,
        contract,
        run_separation=True,
    )
    row = detected.checks.loc[detected.checks["check_id"] == "fixed_effect_separation"]
    assert not row.empty
    assert row.iloc[-1]["status"] == "warn"

    monkeypatch.setattr(
        aow,
        "detect_binary_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic-separation")),
    )
    errored = c.audit_model_readiness_strict(
        data,
        contract,
        run_separation=True,
    )
    row = errored.checks.loc[errored.checks["check_id"] == "fixed_effect_separation"]
    assert row.iloc[-1]["status"] == "warn"

    ddata, dcontract = _duration(4321)
    ddata = ddata.copy()
    ddata.loc[ddata.index[0], "duration"] *= 1000
    duration_audit = c.audit_model_readiness_strict(
        ddata,
        dcontract,
        duration_allowed_range=(1.0, 10_000_000.0),
    )
    assert duration_audit.duration_extremes is not None
    assert duration_audit.duration_boundaries is not None
