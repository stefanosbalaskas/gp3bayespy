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


def _binary(seed=3901):
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
    return sim.data.copy(), contract


def _duration(seed=3902):
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
    return sim.data.copy(), contract


def test_readiness_outcome_identifier_and_full_audit_edges():
    rows, add = _collector()

    r._audit_identifier_types(
        pd.DataFrame({"x": [1]}),
        {
            "participant": None,
            "item": None,
            "trial": None,
        },
        add,
    )
    assert rows[-1][2] == "fail"

    rows.clear()
    unsupported = pd.DataFrame(
        {
            "participant": pd.to_datetime(["2026-01-01"]),
            "item": ["i1"],
            "trial": [1],
        }
    )
    r._audit_identifier_types(
        unsupported,
        {
            "participant": "participant",
            "item": "item",
            "trial": "trial",
        },
        add,
    )
    assert rows[-1][2] == "fail"

    rows.clear()
    r._audit_binary_outcome(pd.Series(["a", "b"]), add)
    assert rows[-1][0] == "outcome_type"

    rows.clear()
    r._audit_binary_outcome(pd.Series([0.0, 2.0]), add)
    assert any(x[0] == "outcome_values" and x[2] == "fail" for x in rows)

    rows.clear()
    r._audit_binary_outcome(pd.Series([0, 0]), add)
    assert rows[-1][0] == "outcome_support"
    assert rows[-1][2] == "fail"

    rows.clear()
    r._audit_duration_outcome(pd.Series([True, False]), add)
    assert rows[-1][0] == "outcome_type"

    rows.clear()
    r._audit_duration_outcome(pd.Series([1.0, np.inf]), add)
    assert any(x[0] == "duration_finite" and x[2] == "fail" for x in rows)

    rows.clear()
    r._audit_duration_outcome(pd.Series([1.0, 0.0]), add)
    assert any(x[0] == "duration_positive" and x[2] == "fail" for x in rows)

    rows.clear()
    r._audit_duration_outcome(pd.Series([1.0, 1.0]), add)
    assert rows[-1][0] == "outcome_support"
    assert rows[-1][2] == "fail"

    bdata, bcontract = _binary()
    empty = r.audit_model_readiness(bdata.iloc[0:0], bcontract)
    assert not empty.ready

    missing = r.audit_model_readiness(
        bdata.drop(columns="trial_covariate"),
        bcontract,
    )
    assert not missing.ready

    missing_values = bdata.copy()
    missing_values.loc[missing_values.index[0], "trial_covariate"] = np.nan
    audit = r.audit_model_readiness(missing_values, bcontract)
    assert not audit.ready

    one_class = bdata.copy()
    one_class["selected"] = 0
    audit = r.audit_model_readiness(one_class, bcontract)
    assert not audit.ready
    assert "Issues:" in repr(audit)

    ddata, dcontract = _duration()
    bad_duration = ddata.copy()
    bad_duration.loc[bad_duration.index[0], "duration"] = 0.0
    assert not r.audit_model_readiness(bad_duration, dcontract).ready

    unsupported_family = replace(bcontract, family="mystery")
    fallback = r.audit_model_readiness(bdata, unsupported_family)
    assert fallback.family == "mystery"
    assert not fallback.ready
    assert "duration_positive" in set(fallback.checks["check_id"])
