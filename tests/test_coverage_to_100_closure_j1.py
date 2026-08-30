from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed: int = 3501, *, condition: bool = True, item: bool = True):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=5,
        trials_per_participant=8,
        n_items=4,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id" if item else None,
        trial_col="trial_id",
        condition_col="condition" if condition else None,
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=False,
    )
    return sim.data.copy(), contract


def _duration(seed: int = 3502):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=5,
        trials_per_participant=8,
        n_items=4,
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


def test_closure_scalar_contract_status_and_balance_matrix():
    assert c._number(1, "x") == 1.0
    with pytest.raises(GP3BayesError):
        c._number(True, "x")
    with pytest.raises(GP3BayesError):
        c._number("x", "x")
    with pytest.raises(GP3BayesError):
        c._number(np.inf, "x")
    with pytest.raises(GP3BayesError):
        c._number(0, "x", lower=0, lower_open=True)
    with pytest.raises(GP3BayesError):
        c._number(2, "x", upper=2, upper_open=True)
    with pytest.raises(GP3BayesError):
        c._integer(1.5, "x")
    assert c._integer(2, "x") == 2

    _, bcontract = _binary()
    with pytest.raises(GP3BayesError):
        c._contract(object())
    with pytest.raises(GP3BayesError):
        c._contract(bcontract, "duration")

    with pytest.raises(GP3BayesError):
        c._closure_fixed_model_matrix(
            pd.DataFrame({"x": [1]}),
            SimpleNamespace(family="pupil"),
        )

    assert c._worst_status(["pass", "warn"]) == "review"
    assert c._worst_status(["pass", "not_applicable"]) == "pass"
    assert c._worst_status(["pass", "fail"]) == "fail"

    data, no_condition = _binary(3510, condition=False)
    balance = c.summarise_condition_balance(data, no_condition)
    assert balance.status == "not_applicable"

    data, contract = _binary(3511)
    with pytest.raises(GP3BayesError):
        c.summarise_condition_balance(
            data.drop(columns=["condition"]),
            contract,
        )

    empty = data.iloc[0:0].copy()
    balance = c.summarise_condition_balance(empty, contract)
    assert balance.status == "fail"

    one = data.copy()
    one["condition"] = "control"
    balance = c.summarise_condition_balance(one, contract)
    assert balance.status == "fail"

    imbalanced = data.copy()
    imbalanced["condition"] = "control"
    imbalanced.loc[imbalanced.index[:2], "condition"] = "treatment"
    balance = c.summarise_condition_balance(
        imbalanced,
        contract,
        warning_fraction=0.10,
        failure_fraction=0.01,
    )
    assert balance.status == "review"


def test_closure_group_variation_identifier_and_duration_edge_matrix():
    data, no_item = _binary(3520, item=False)
    item = c.summarise_binary_group_variation(data, no_item, group="item")
    assert item.status == "not_applicable"

    _, contract = _binary(3521)
    with pytest.raises(GP3BayesError):
        c.summarise_binary_group_variation(
            data,
            contract,
            group="bad",
        )

    all_zero = data.copy()
    all_zero["selected"] = 0
    variation = c.summarise_binary_group_variation(all_zero, contract, group="participant")
    assert variation.status == "fail"

    mixed = data.copy()
    first = mixed["participant_id"].iloc[0]
    mixed.loc[mixed["participant_id"] == first, "selected"] = 0
    variation = c.summarise_binary_group_variation(mixed, contract, group="participant")
    assert variation.status in {"review", "fail"}

    with pytest.raises(GP3BayesError):
        c.identify_identifier_like_predictors(
            data.drop(columns=["trial_covariate"]),
            contract,
        )

    identifier_data = data.copy()
    identifier_data["participant_covariate"] = np.arange(len(identifier_data), dtype=float)
    identifier = c.identify_identifier_like_predictors(
        identifier_data,
        contract,
        unique_fraction=0.5,
    )
    assert "participant_covariate" in identifier.table["predictor"].values

    text_data = data.copy()
    text_data["participant_covariate"] = [f"id-{i}" for i in range(len(text_data))]
    identifier = c.identify_identifier_like_predictors(
        text_data,
        contract,
        unique_fraction=0.5,
    )
    assert not identifier.table.empty

    ddata, dcontract = _duration()
    bad = ddata.copy()
    bad.loc[bad.index[0], "duration"] = 0.0
    with pytest.raises(GP3BayesError):
        c.review_duration_extremes(bad, dcontract)

    constant = ddata.copy()
    constant["duration"] = 500.0
    extreme = c.review_duration_extremes(constant, dcontract)
    assert extreme.n_flagged == 0

    outlier = ddata.copy()
    outlier.loc[outlier.index[0], "duration"] *= 1000
    extreme = c.review_duration_extremes(
        outlier,
        dcontract,
        mad_cutoff=2,
        iqr_multiplier=1,
    )
    assert extreme.n_flagged >= 1

    assert c._censor_flags(pd.Series([True, False])).tolist() == [True, False]
    assert c._censor_flags(pd.Series([0, 2])).tolist() == [False, True]
    assert c._censor_flags(pd.Series(["yes", "no", "right"])).tolist() == [True, False, True]


def test_duration_boundary_and_strict_readiness_branch_matrix():
    data, contract = _duration(3530)

    with pytest.raises(GP3BayesError):
        c.audit_duration_boundaries(
            data,
            contract,
            allowed_range=(1000, 10),
        )
    with pytest.raises(GP3BayesError):
        c.audit_duration_boundaries(
            data,
            contract,
            censor_col="missing",
        )

    no_detect = c.audit_duration_boundaries(
        data,
        contract,
        detect_candidate_columns=False,
    )
    assert "not_applicable" in set(no_detect.checks["status"])

    candidates = data.copy()
    candidates["censor_flag"] = 0
    review = c.audit_duration_boundaries(candidates, contract)
    assert "censor_flag" in review.candidate_columns
    assert "warn" in set(review.checks["status"])

    censored = data.copy()
    censored["censor"] = ["yes"] + ["no"] * (len(censored) - 1)
    audit = c.audit_duration_boundaries(
        censored,
        contract,
        censor_col="censor",
    )
    assert audit.censored_rows == (1,)
    assert audit.status == "fail"

    within = c.audit_duration_boundaries(
        data,
        contract,
        allowed_range=(
            float(data["duration"].min()) * 0.5,
            float(data["duration"].max()) * 1.5,
        ),
    )
    assert not within.range_violations

    violated = c.audit_duration_boundaries(
        data,
        contract,
        allowed_range=(1.0, 2.0),
    )
    assert violated.range_violations

    bdata, bcontract = _binary(3531)
    strict_b = c.audit_model_readiness_strict(
        bdata,
        bcontract,
        run_separation=False,
    )
    assert strict_b.family == "binary"
    assert strict_b.separation is None

    strict_d = c.audit_model_readiness_strict(
        data,
        contract,
        run_separation=False,
        duration_allowed_range=(
            float(data["duration"].min()) * 0.5,
            float(data["duration"].max()) * 1.5,
        ),
    )
    assert strict_d.family == "duration"
    assert strict_d.duration_extremes is not None
    assert strict_d.duration_boundaries is not None
