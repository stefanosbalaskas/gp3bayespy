from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError

c = importlib.import_module("gp3bayespy.specification_closure")


def _duration_contract():
    return gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="p",
        condition_col="condition",
        outcome_unit="milliseconds",
    )


def _binary_data_contract(seed=5401):
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
    )
    return sim.data, contract


def test_closure_identifier_empty_nan_constant_and_duration_extreme_paths():
    data = pd.DataFrame({"y": [0, 1], "p": ["p1", "p2"]})
    contract = gp.create_model_contract(family="binary", outcome_col="y", participant_col="p")
    empty = c.identify_identifier_like_predictors(data, contract)
    assert empty.flagged == ()
    assert empty.table.empty

    nan_data = data.assign(x=np.nan)
    nan_contract = gp.create_model_contract(
        family="binary",
        outcome_col="y",
        participant_col="p",
        predictors=("x",),
    )
    nan_audit = c.identify_identifier_like_predictors(nan_data, nan_contract)
    assert nan_audit.table.iloc[0]["reason"] == "none"

    constant_data = data.assign(x=1.0)
    constant_audit = c.identify_identifier_like_predictors(
        constant_data,
        nan_contract,
        unique_fraction=0.1,
        integer_fraction=0.5,
        monotone_correlation=0.5,
    )
    assert np.isnan(constant_audit.table.iloc[0]["row_order_correlation"])

    dcontract = _duration_contract()
    with pytest.raises(GP3BayesError):
        c.review_duration_extremes(pd.DataFrame({"p": ["p1"]}), dcontract)
    with pytest.raises(GP3BayesError):
        c.review_duration_extremes(
            pd.DataFrame(
                {
                    "duration": [1.0, 0.0],
                    "p": ["p1", "p1"],
                    "condition": [0, 1],
                }
            ),
            dcontract,
        )
    extreme = c.review_duration_extremes(
        pd.DataFrame(
            {
                "duration": [1.0] * 9 + [1e20],
                "p": ["p1"] * 10,
                "condition": [0, 1] * 5,
            }
        ),
        dcontract,
    )
    assert extreme.n_flagged >= 1


def test_closure_censor_flag_types_and_boundary_matrix():
    assert c._censor_flags(pd.Series([True, False])).tolist() == [True, False]
    assert c._censor_flags(pd.Series([0, 2])).tolist() == [False, True]
    assert c._censor_flags(pd.Series(["no", "censored"])).tolist() == [False, True]

    contract = _duration_contract()
    data = pd.DataFrame(
        {
            "duration": [1.0, 2.0, 100.0],
            "p": ["p1", "p1", "p2"],
            "condition": [0, 1, 0],
        }
    )
    with pytest.raises(GP3BayesError):
        c.audit_duration_boundaries(data, contract, allowed_range=(0, 10))
    audited = c.audit_duration_boundaries(data, contract, allowed_range=(1, 10))
    assert audited.range_violations == (3,)

    with pytest.raises(GP3BayesError):
        c.audit_duration_boundaries(data, contract, censor_col="missing")

    censored = data.assign(censored=[False, True, False])
    audit2 = c.audit_duration_boundaries(censored, contract, censor_col="censored")
    assert audit2.censored_rows == (2,)

    candidate = data.assign(deadline_flag=False)
    audit3 = c.audit_duration_boundaries(candidate, contract)
    assert "deadline_flag" in audit3.candidate_columns


def test_closure_strict_mapping_separation_and_base_check_fallbacks(monkeypatch):
    data, contract = _binary_data_contract()
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")

    monkeypatch.setattr(
        aow,
        "detect_binary_separation",
        lambda *a, **k: {"separation_detected": False},
    )
    audit = c.audit_model_readiness_strict(data, contract, run_separation=True)
    sep = audit.checks.loc[audit.checks["check_id"].eq("fixed_effect_separation")]
    assert sep.iloc[-1]["status"] == "pass"

    original = c.audit_model_readiness
    monkeypatch.setattr(
        c,
        "audit_model_readiness",
        lambda *a, **k: SimpleNamespace(checks="not-a-table"),
    )
    fallback = c.audit_model_readiness_strict(data, contract, run_separation=False)
    assert not fallback.checks.empty

    monkeypatch.setattr(
        c,
        "audit_model_readiness",
        lambda *a, **k: SimpleNamespace(checks=pd.DataFrame({"unrelated": [1]})),
    )
    no_common = c.audit_model_readiness_strict(data, contract, run_separation=False)
    assert "status" in no_common.checks

    monkeypatch.setattr(c, "audit_model_readiness", original)


def _interaction_specs(seed=5410):
    b = importlib.import_module("gp3bayespy.binary")
    d = importlib.import_module("gp3bayespy.duration")
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")

    bsim = gp.simulate_hierarchical_binary_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed
    )
    bcontract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("participant_covariate", "trial_covariate"),
    )
    bprepared = gp.prepare_hierarchical_binary_data(bsim.data, bcontract)
    bspec = b.specify_binary_model(bprepared)
    badv = aow.InteractionPriorSpecification(
        base=bspec,
        advanced_priors={
            "main_effect_scale": 0.5,
            "interaction_scale": 0.25,
            "interaction": bcontract.interaction,
        },
    )

    dsim = gp.simulate_hierarchical_duration_data(
        n_participants=4, trials_per_participant=6, n_items=3, seed=seed + 1
    )
    dcontract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        interaction=("participant_covariate", "trial_covariate"),
        outcome_unit="milliseconds",
    )
    dprepared = gp.prepare_hierarchical_duration_data(dsim.data, dcontract)
    dspec = d.specify_duration_model(dprepared, baseline=500.0)
    dadv = aow.InteractionPriorSpecification(
        base=dspec,
        advanced_priors={
            "main_effect_scale": 0.3,
            "interaction_scale": 0.2,
            "interaction": dcontract.interaction,
        },
    )
    return (bprepared, badv), (dprepared, dadv)


def test_closure_rebuild_advanced_interaction_branches(monkeypatch):
    aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
    (bprepared, badv), (dprepared, dadv) = _interaction_specs()

    binary_token = object()
    duration_token = object()
    monkeypatch.setattr(
        aow,
        "specify_binary_model_with_interaction_prior",
        lambda *a, **k: binary_token,
    )
    monkeypatch.setattr(
        aow,
        "specify_duration_model_with_interaction_prior",
        lambda *a, **k: duration_token,
    )

    assert c._rebuild(bprepared, badv) is binary_token
    assert c._rebuild(dprepared, dadv) is duration_token
