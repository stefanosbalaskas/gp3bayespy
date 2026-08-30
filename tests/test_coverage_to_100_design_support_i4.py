from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import gp3bayespy as gp
import gp3bayespy.design_support_diagnostics as d
from gp3bayespy.exceptions import GP3BayesError


def _binary(seed: int = 3101, *, random_slope: bool = False):
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
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=random_slope,
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    return sim.data.copy(), contract, prepared


def test_design_input_missingness_and_threshold_branches():
    data, contract, prepared = _binary()

    with pytest.raises(GP3BayesError):
        d._input(data)
    with pytest.raises(GP3BayesError):
        d._input(SimpleNamespace(contract=contract, data=[1, 2]))
    with pytest.raises(GP3BayesError):
        d._input(object())
    with pytest.raises(GP3BayesError):
        d._input(
            SimpleNamespace(
                specification=SimpleNamespace(
                    prepared=SimpleNamespace(data=[1, 2]),
                    contract=contract,
                )
            )
        )

    via_prepared = d._input(prepared)
    assert len(via_prepared[0]) == len(prepared.data)

    interaction_contract = replace(
        contract,
        interaction=("condition", "trial_covariate"),
    )
    declared = d._declared_columns(interaction_contract)
    assert "condition" in declared
    assert "trial_covariate" in declared

    with pytest.raises(GP3BayesError):
        d.audit_missingness_structure(
            data,
            contract,
            review_fraction=0.3,
            fail_fraction=0.2,
        )

    empty = data.iloc[0:0].copy()
    empty_audit = d.audit_missingness_structure(empty, contract)
    assert empty_audit.status == "fail"

    absent = d.audit_missingness_structure(
        data.drop(columns="trial_covariate"),
        contract,
    )
    assert absent.status == "fail"
    assert "trial_covariate" in absent.absent_columns

    mixed = data.copy()
    mixed.loc[mixed.index[:2], "trial_covariate"] = np.nan
    review = d.audit_missingness_structure(
        mixed,
        contract,
        review_fraction=0.01,
        fail_fraction=0.20,
    )
    assert "review" in set(review.column_table["status"])

    failed = d.audit_missingness_structure(
        mixed,
        contract,
        review_fraction=0.01,
        fail_fraction=0.02,
    )
    assert "fail" in set(failed.column_table["status"])
    assert not failed.grouping_table.empty


def test_fixed_random_and_separation_branch_matrix(monkeypatch):
    data, contract, prepared = _binary(3110)

    with pytest.raises(GP3BayesError):
        d.audit_fixed_effect_design(
            data,
            contract,
            condition_number_review=0.5,
        )
    with pytest.raises(GP3BayesError):
        d.audit_fixed_effect_design(
            data,
            contract,
            condition_number_review=10,
            condition_number_fail=5,
        )
    with pytest.raises(GP3BayesError):
        d.audit_fixed_effect_design(
            data,
            contract,
            leverage_multiplier=0.5,
        )

    all_missing = data.copy()
    all_missing["trial_covariate"] = np.nan
    empty = d.audit_fixed_effect_design(all_missing, contract)
    assert empty.status == "fail"
    assert empty.error is not None

    monkeypatch.setattr(
        d,
        "_fixed_model_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("matrix failure")),
    )
    failed = d.audit_fixed_effect_design(data, contract)
    assert failed.status == "fail"
    assert "matrix failure" in str(failed.error)

    monkeypatch.setattr(
        d,
        "_fixed_model_matrix",
        lambda frame, model_contract: (
            np.zeros((len(frame), 1), dtype=float),
            ["Intercept"],
        ),
    )
    rank_zero = d.audit_fixed_effect_design(data, contract)
    assert rank_zero.rank == 0
    assert rank_zero.status == "fail"

    monkeypatch.undo()

    with pytest.raises(GP3BayesError):
        d.audit_random_effects_support(
            data,
            contract,
            minimum_repeated_rows=0,
        )

    missing_participant = d.audit_random_effects_support(
        data.drop(columns="participant_id"),
        contract,
    )
    assert missing_participant.status == "fail"
    assert missing_participant.error is not None

    one = data.loc[data["participant_id"] == data["participant_id"].iloc[0]].copy()
    one_support = d.audit_random_effects_support(one, contract)
    assert one_support.status == "review"

    sparse_item = data.copy()
    sparse_item["item_id"] = [f"unique-{i}" for i in range(len(sparse_item))]
    item_support = d.audit_random_effects_support(sparse_item, contract)
    assert "review" in set(item_support.component_table["status"])

    _, slope_contract, _ = _binary(3115, random_slope=True)
    missing_condition = d.audit_random_effects_support(
        data.drop(columns="condition"),
        slope_contract,
    )
    assert "fail" in set(missing_condition.component_table["status"])

    one_condition = data.copy()
    one_condition["condition"] = "control"
    slope_fail = d.audit_random_effects_support(
        one_condition,
        slope_contract,
    )
    assert "fail" in set(slope_fail.component_table["status"])

    original = d._fixed_model_matrix
    monkeypatch.setattr(
        d,
        "_fixed_model_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("separation screen failure")),
    )
    separation = d.audit_design_support(
        data,
        contract,
        separation=True,
        strict_readiness=False,
    )
    assert separation.separation["status"] == "review"
    assert "separation screen failure" in separation.separation["detail"]
    assert separation.strict_readiness is None

    monkeypatch.setattr(d, "_fixed_model_matrix", original)
    preflight = d.preflight_model_specification(
        prepared,
        separation=False,
        strict_readiness=True,
    )
    assert preflight.component_table.shape[0] == 6
