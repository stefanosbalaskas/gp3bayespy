from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.specification_closure as c
from gp3bayespy.exceptions import GP3BayesError


def _estimand(seed: int, family: str = "binary") -> c.Estimand:
    draws = np.linspace(-0.15, 0.15, 40) + seed * 0.001
    quantity = "probability_difference" if family == "binary" else "conditional_median_ratio"
    return c.Estimand(
        family,
        quantity,
        pd.DataFrame(
            {
                ".draw": np.arange(1, 41),
                quantity: draws,
            }
        ),
        {"seed": seed},
    )


def _plan_spec():
    data = pd.DataFrame(
        {
            "participant_id": [
                "p1",
                "p1",
                "p2",
                "p2",
                "p3",
                "p3",
            ],
            "y": [0, 1, 0, 1, 0, 1],
        }
    )
    prepared = SimpleNamespace(data=data)
    return SimpleNamespace(
        family="binary",
        prepared=prepared,
        contract=object(),
    )


def test_group_deletion_run_success_error_and_retained_fits(monkeypatch):
    spec = _plan_spec()
    plan = c.GroupDeletionSensitivityPlan(
        "0.2",
        "binary",
        "participant",
        "participant_id",
        ("p1", "p2"),
        pd.DataFrame(),
        spec,
        20,
    )

    def fake_fit_spec(
        specification,
        backend,
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    ):
        if seed == 12:
            raise RuntimeError("synthetic refit failure")
        return SimpleNamespace(
            family="binary",
            specification=specification,
            seed=seed,
        )

    def fake_primary(fit, ndraws, seed):
        return _estimand(seed)

    monkeypatch.setattr(c, "_fit_spec", fake_fit_spec)
    monkeypatch.setattr(c, "_primary_estimand", fake_primary)
    monkeypatch.setattr(
        c,
        "invert_transformation_recipe",
        lambda data, prepared: data.copy(),
    )
    monkeypatch.setattr(
        c,
        "_reprepare",
        lambda specification, contract, raw: SimpleNamespace(data=raw.copy()),
    )
    monkeypatch.setattr(
        c,
        "_rebuild",
        lambda prepared, template: SimpleNamespace(
            family="binary",
            prepared=prepared,
        ),
    )

    run = c.run_group_deletion_sensitivity(
        plan,
        backend="analytic",
        chains=1,
        iter=100,
        warmup=20,
        cores=1,
        seed=10,
        ndraws=30,
        retain_fits=True,
    )
    assert run.reference_fit is not None
    assert set(run.summary["status"]) == {"completed", "error"}
    assert "p1" in run.results
    assert "p2" not in run.results
    assert run.fits is not None
    assert "p1" in run.fits

    with pytest.raises(GP3BayesError):
        c.run_group_deletion_sensitivity(object())  # type: ignore[arg-type]


def test_random_slope_run_both_retention_paths_and_primary_dispatch(monkeypatch):
    original_primary = c._primary_estimand
    fit_a = SimpleNamespace(family="binary", name="a")
    fit_b = SimpleNamespace(family="binary", name="b")
    node_a = {"ready": True, "specification": fit_a}
    node_b = {"ready": True, "specification": fit_b}
    plan = c.RandomSlopeSensitivityPlan(
        "0.2",
        "binary",
        node_a,
        node_b,
    )

    def fake_fit_spec(
        specification,
        backend,
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    ):
        return SimpleNamespace(
            family="binary",
            source=specification,
            seed=seed,
        )

    monkeypatch.setattr(c, "_fit_spec", fake_fit_spec)
    monkeypatch.setattr(
        c,
        "_primary_estimand",
        lambda fit, ndraws, seed: _estimand(seed),
    )

    retained = c.run_random_slope_sensitivity(
        plan,
        backend="analytic",
        chains=1,
        iter=100,
        warmup=20,
        cores=1,
        seed=21,
        retain_fits=True,
    )
    assert retained["fits"] is not None
    assert retained["automatic_selection"] is False
    assert set(retained["estimands"]) == {"random_intercept", "random_slope"}

    discarded = c.run_random_slope_sensitivity(
        plan,
        backend="analytic",
        chains=1,
        iter=100,
        warmup=20,
        cores=1,
        seed=31,
        retain_fits=False,
    )
    assert discarded["fits"] is None

    with pytest.raises(GP3BayesError):
        c.run_random_slope_sensitivity(object())  # type: ignore[arg-type]

    bad_plan = c.RandomSlopeSensitivityPlan(
        "0.2",
        "binary",
        {"ready": False, "specification": None},
        node_b,
    )
    with pytest.raises(GP3BayesError):
        c.run_random_slope_sensitivity(bad_plan)

    primary = original_primary
    monkeypatch.setattr(
        c,
        "estimate_standardized_probability_contrast",
        lambda fit, ndraws=None: "binary-primary",
    )
    monkeypatch.setattr(
        c,
        "estimate_standardized_duration_estimands",
        lambda fit, ndraws=None, seed=1: "duration-primary",
    )
    assert primary(SimpleNamespace(family="binary"), None, 1) == "binary-primary"
    assert primary(SimpleNamespace(family="duration"), None, 1) == "duration-primary"


def test_group_deletion_plan_contract_errors_and_unit_selection(monkeypatch):
    spec = _plan_spec()
    spec.contract = SimpleNamespace(
        mappings={
            "participant": "participant_id",
            "item": None,
        }
    )

    monkeypatch.setattr(
        c,
        "_mapping",
        lambda contract, key: contract.mappings.get(key),
    )
    monkeypatch.setattr(
        c,
        "audit_model_readiness",
        lambda data, contract: SimpleNamespace(
            ready=len(data) >= 2,
            status="pass" if len(data) >= 2 else "review",
        ),
    )

    plan = c.create_group_deletion_sensitivity_plan(
        spec,
        group="participant",
        units=("p1", "p3"),
        max_units=3,
    )
    assert plan.units == ("p1", "p3")
    assert len(plan.table) == 2

    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(spec, group="bad")
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(spec, group="item")
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(
            spec,
            group="participant",
            units=("p1", "p1"),
        )
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(
            spec,
            group="participant",
            units=("unknown",),
        )
    with pytest.raises(GP3BayesError):
        c.create_group_deletion_sensitivity_plan(
            spec,
            group="participant",
            max_units=2,
        )
