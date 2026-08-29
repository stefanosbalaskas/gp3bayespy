from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.advanced_optional_workflows as aow
import gp3bayespy.predictive as predictive
import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _binary_contract(*, condition: bool = True, item: bool = True):
    return gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id" if item else None,
        trial_col="trial_id",
        condition_col="condition" if condition else None,
        predictors=("x",),
        interaction=None,
        random_slope=False,
    )


def _duration_contract():
    return gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("x",),
        interaction=None,
        random_slope=False,
        outcome_unit="milliseconds",
    )


def _binary_data():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "item_id": ["i1", "i2", "i1", "i2", "i1", "i2"],
            "trial_id": np.arange(6),
            "condition": [0, 1, 0, 1, 0, 1],
            "x": np.linspace(-1, 1, 6),
            "selected": [0, 1, 0, 1, 1, 1],
        }
    )


def _duration_data():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "item_id": ["i1", "i2", "i1", "i2", "i1", "i2"],
            "trial_id": np.arange(6),
            "condition": [0, 1, 0, 1, 0, 1],
            "x": np.linspace(-1, 1, 6),
            "duration": [300, 450, 500, 650, 800, 1000],
        }
    )


def test_closure_scalar_contract_status_and_matrix_helpers():
    assert c._number(0.5, "x", 0, 1) == 0.5
    assert c._integer(3.0, "n", 2) == 3
    for value in (True, "x", np.inf):
        with pytest.raises(GP3BayesError):
            c._number(value, "x")
    with pytest.raises(GP3BayesError):
        c._number(0, "x", lower=0, lower_open=True)
    with pytest.raises(GP3BayesError):
        c._number(1, "x", upper=1, upper_open=True)
    with pytest.raises(GP3BayesError):
        c._integer(1.5, "n", 0)
    with pytest.raises(GP3BayesError):
        c._integer(1, "n", 2)

    bcontract = _binary_contract()
    assert c._contract(bcontract) is bcontract
    assert c._contract(bcontract, "binary") is bcontract
    with pytest.raises(GP3BayesError):
        c._contract(object())
    with pytest.raises(GP3BayesError):
        c._contract(bcontract, "duration")
    assert c._mapping(bcontract, "outcome") == "selected"

    row = c._check_row("id", "domain", "pass", "ok", 3)
    assert row["n"] == 3
    assert c._worst_status(["pass"]) == "pass"
    assert c._worst_status(["review", "pass"]) == "review"
    assert c._worst_status(["warn"]) == "review"
    assert c._worst_status(["fail", "pass"]) == "fail"
    assert c._worst_status(["not_applicable"]) == "not_applicable"

    with pytest.raises(GP3BayesError):
        c._closure_fixed_model_matrix(pd.DataFrame(), SimpleNamespace(family="other"))


def test_condition_balance_branches_and_validation():
    data = _binary_data()
    contract = _binary_contract()
    passed = c.summarise_condition_balance(data, contract)
    assert passed.status == "pass"
    assert np.isclose(passed.minimum_fraction, 0.5)

    no_condition = c.summarise_condition_balance(data, _binary_contract(condition=False))
    assert no_condition.status == "not_applicable"

    empty = data.copy()
    empty["condition"] = np.nan
    failed = c.summarise_condition_balance(empty, contract)
    assert failed.status == "fail"

    one_level = data.copy()
    one_level["condition"] = 0
    assert c.summarise_condition_balance(one_level, contract).status == "fail"

    imbalanced = pd.concat([data.iloc[[0]]] * 20 + [data.iloc[[1]]], ignore_index=True)
    review_or_fail = c.summarise_condition_balance(
        imbalanced,
        contract,
        warning_fraction=0.10,
        failure_fraction=0.02,
    )
    assert review_or_fail.status in {"review", "pass", "fail"}

    with pytest.raises(GP3BayesError):
        c.summarise_condition_balance(data.drop(columns="condition"), contract)
    with pytest.raises(GP3BayesError):
        c.summarise_condition_balance(data, contract, warning_fraction=0)
    with pytest.raises(GP3BayesError):
        c.summarise_condition_balance(
            data,
            contract,
            warning_fraction=0.1,
            failure_fraction=0.1,
        )


def test_binary_group_variation_pass_review_not_applicable_and_errors():
    data = _binary_data()
    contract = _binary_contract()
    participant = c.summarise_binary_group_variation(data, contract, group="participant")
    assert participant.group == "participant"
    assert participant.n_no_variation >= 1

    item = c.summarise_binary_group_variation(data, contract, group="item")
    assert item.group == "item"

    no_item = c.summarise_binary_group_variation(
        data,
        _binary_contract(item=False),
        group="item",
    )
    assert no_item.status == "not_applicable"

    with pytest.raises(GP3BayesError):
        c.summarise_binary_group_variation(data, contract, group="bad")
    with pytest.raises(GP3BayesError):
        c.summarise_binary_group_variation(
            data.drop(columns="selected"),
            contract,
            group="participant",
        )


def test_detailed_binary_ppc_groups_and_family_guard(monkeypatch):
    data = _binary_data()
    contract = _binary_contract()
    prepared = SimpleNamespace(data=data, contract=contract)
    fit = SimpleNamespace(
        family="binary",
        specification=SimpleNamespace(prepared=prepared, contract=contract),
    )
    rng = np.random.default_rng(1401)
    pred = SimpleNamespace(
        observed=data["selected"],
        draws=rng.binomial(
            1,
            np.broadcast_to(
                np.linspace(0.2, 0.8, len(data))[None, :],
                (40, len(data)),
            ),
        ).astype(float),
    )
    expected = SimpleNamespace(
        observed=data["selected"],
        draws=np.broadcast_to(
            np.linspace(0.2, 0.8, len(data))[None, :],
            (40, len(data)),
        ),
    )
    monkeypatch.setattr(
        predictive,
        "predict_model",
        lambda fit, type="expected", **kwargs: pred if type == "predictive" else expected,
    )
    monkeypatch.setattr(
        predictive,
        "binary_calibration_table",
        lambda x, bins=10: pd.DataFrame(
            {"bin": [1], "observed_rate": [0.5], "mean_predicted_probability": [0.5]}
        ),
    )
    result = c.check_binary_ppc_details(
        fit,
        draws=40,
        calibration_bins=4,
        sparse_cell_min=3,
    )
    assert result["family"] == "binary"
    assert set(result["groups"]) == {"participant", "item"}
    assert "sparse" in result["groups"]["participant"]

    with pytest.raises(GP3BayesError):
        c.check_binary_ppc_details(SimpleNamespace(family="duration"))


def test_detailed_duration_ppc_quantiles_tail_and_family_guard(monkeypatch):
    data = _duration_data()
    contract = _duration_contract()
    prepared = SimpleNamespace(data=data, contract=contract)
    fit = SimpleNamespace(
        family="duration",
        specification=SimpleNamespace(prepared=prepared, contract=contract),
    )
    rng = np.random.default_rng(1402)
    pred = SimpleNamespace(
        observed=data["duration"],
        draws=np.maximum(
            rng.normal(
                data["duration"].to_numpy()[None, :],
                50,
                size=(50, len(data)),
            ),
            1,
        ),
    )
    monkeypatch.setattr(predictive, "predict_model", lambda *args, **kwargs: pred)

    automatic = c.check_duration_ppc_details(
        fit,
        draws=50,
        quantiles=(0.5, 0.9),
    )
    explicit = c.check_duration_ppc_details(
        fit,
        draws=50,
        quantiles=(0.25, 0.75),
        tail_threshold=700,
    )
    assert len(automatic["quantiles"]) == 2
    assert automatic["tail_threshold"] > 0
    assert explicit["tail_threshold"] == 700

    with pytest.raises(GP3BayesError):
        c.check_duration_ppc_details(SimpleNamespace(family="binary"))


def test_kfold_random_grouped_stratified_and_errors(monkeypatch):
    data = _binary_data()
    contract = _binary_contract()
    prepared = SimpleNamespace(data=data)
    fit = SimpleNamespace(
        fit_performed=True,
        family="binary",
        specification=SimpleNamespace(prepared=prepared, contract=contract),
    )
    pointwise = pd.DataFrame({"elpd_loo": [-1.0, -1.1, -0.9, -1.2, -1.05, -0.95]})
    monkeypatch.setattr(
        aow,
        "compute_psis_loo",
        lambda fit: SimpleNamespace(pointwise=pointwise),
    )

    random = c.compute_kfold_cv(fit, K=3, folds="random", seed=1)
    grouped = c.compute_kfold_cv(
        fit,
        K=3,
        folds="grouped",
        group="participant_id",
        seed=1,
    )
    stratified = c.compute_kfold_cv(fit, K=2, folds="stratified", seed=1)
    assert random.K == 3
    assert grouped.folds == "grouped"
    assert stratified.folds == "stratified"
    assert np.isfinite(random.total_elpd)
    assert not random.automatic_selection

    one = pointwise.iloc[:1]
    monkeypatch.setattr(
        aow,
        "compute_psis_loo",
        lambda fit: SimpleNamespace(pointwise=one),
    )
    one_fit = SimpleNamespace(
        fit_performed=True,
        family="binary",
        specification=SimpleNamespace(
            prepared=SimpleNamespace(data=data.iloc[:1].copy()),
            contract=contract,
        ),
    )
    one_result = c.compute_kfold_cv(one_fit, K=2)
    assert np.isnan(one_result.se_elpd)

    with pytest.raises(GP3BayesError):
        c.compute_kfold_cv(SimpleNamespace(fit_performed=False, family="binary"))
    with pytest.raises(GP3BayesError):
        c.compute_kfold_cv(fit, K=1)
    with pytest.raises(GP3BayesError):
        c.compute_kfold_cv(fit, folds="bad")
    with pytest.raises(GP3BayesError):
        c.compute_kfold_cv(fit, folds="grouped")
    with pytest.raises(GP3BayesError):
        c.compute_kfold_cv(fit, folds="grouped", group="missing")


def test_specification_traceability_and_sensitivity_plan_guards():
    trace = c.gp3bayes_specification_traceability()
    assert len(trace) >= 10
    assert not trace["automatic_decision"].any()

    with pytest.raises(GP3BayesError):
        c.run_group_deletion_sensitivity(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        c.run_random_slope_sensitivity(object())  # type: ignore[arg-type]

    # Construct minimal real RandomSlopeSensitivityPlan to hit readiness guard.
    plan = c.RandomSlopeSensitivityPlan(
        plan_version="0.2",
        family="binary",
        intercept_only={"ready": False},
        random_slope={"ready": True},
    )
    with pytest.raises(GP3BayesError, match="readiness"):
        c.run_random_slope_sensitivity(plan)
