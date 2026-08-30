from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.predictive as p
from gp3bayespy.exceptions import GP3BayesError


def _arr(values):
    return SimpleNamespace(values=np.asarray(values, dtype=float))


def _fit(seed: int = 3601):
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
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    spec = gp.specify_binary_model(prepared)
    draws = 8
    pcols = len(prepared.model_matrix_columns)
    npart = prepared.data["participant_id"].nunique()
    nitem = prepared.data["item_id"].nunique()
    posterior = {
        "b_Intercept": _arr(np.zeros((1, draws))),
        "b": _arr(np.zeros((1, draws, max(pcols - 1, 0)))),
        "sd_participant": _arr(np.full((1, draws), 0.2)),
        "participant_z": _arr(np.zeros((1, draws, npart))),
        "sd_item": _arr(np.full((1, draws), 0.1)),
        "item_z": _arr(np.zeros((1, draws, nitem))),
    }
    fit = binary.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=SimpleNamespace(formula_text="selected ~ condition"),
        backend_fit=SimpleNamespace(posterior=posterior),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )
    return fit


def test_predictive_scalar_probability_and_object_guard_matrix():
    assert p._flag(True, "x")
    with pytest.raises(GP3BayesError):
        p._flag(1, "x")

    assert p._positive_integer_or_none(None, "x") is None
    assert p._positive_integer_or_none(2, "x") == 2
    for value in (True, 0, 1.5, float("inf"), "2"):
        with pytest.raises(GP3BayesError):
            p._positive_integer_or_none(value, "x")

    assert p._nonnegative_integer(0, "x") == 0
    for value in (True, -1, 1.5, float("nan"), "1"):
        with pytest.raises(GP3BayesError):
            p._nonnegative_integer(value, "x")

    with pytest.raises(GP3BayesError):
        p._positive_integer(None, "x")

    assert p._probabilities((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    for probs in (
        ("x", 0.5, 0.9),
        (0.1, 0.9),
        (0.1, 0.1, 0.9),
        (-0.1, 0.5, 0.9),
        (0.1, 0.5, float("nan")),
    ):
        with pytest.raises(GP3BayesError):
            p._probabilities(probs)

    fit = _fit()
    with pytest.raises(GP3BayesError):
        p._validate_fit(fit, "duration")
    bad_fit = replace(
        fit,
        backend_fit=SimpleNamespace(posterior=None),
    )
    with pytest.raises(GP3BayesError):
        p._validate_fit(bad_fit)

    with pytest.raises(GP3BayesError):
        p._object_parts(object())

    no_prepared = replace(fit.specification, prepared=None)
    with pytest.raises(GP3BayesError):
        p._object_parts(no_prepared)

    empty_prepared = replace(
        fit.specification.prepared,
        data=fit.specification.prepared.data.iloc[0:0].copy(),
    )
    empty_spec = replace(fit.specification, prepared=empty_prepared)
    with pytest.raises(GP3BayesError):
        p._object_parts(empty_spec)


def test_predictive_type_restore_representative_and_grid_matrix():
    assert p._as_values(pd.Series([1, 2])) == [1, 2]
    assert p._as_values(pd.Index(["a", "b"])) == ["a", "b"]
    assert p._as_values(np.asarray([1, 2])) == [1, 2]
    assert p._as_values(np.asarray(3)) == [3]
    assert p._as_values(("a", "b")) == ["a", "b"]
    assert p._as_values("a") == ["a"]

    categorical = pd.Series(pd.Categorical(["b", "a"], categories=["a", "b"]))
    assert list(p._restore_type(["a"], categorical).cat.categories) == ["a", "b"]
    assert p._restore_type([1, 2], pd.Series([1, 2], dtype="int64")).dtype.kind in "iu"
    assert p._restore_type([True], pd.Series([True], dtype=bool)).dtype == bool
    assert p._restore_type(["x"], pd.Series(["a"], dtype=object)).dtype == object
    assert p._restore_type([1.5], pd.Series([1.0], dtype=float)).dtype.kind == "f"
    dt = pd.Series(pd.to_datetime(["2026-01-01"]))
    assert len(p._restore_type([dt.iloc[0]], dt)) == 1

    assert p._representative_value(categorical, "median") == "a"
    assert not bool(p._representative_value(pd.Series([False, True]), "median"))
    assert p._representative_value(pd.Series([1.0, 3.0]), "median") == 2.0
    assert p._representative_value(pd.Series([1.0, 3.0]), "mean") == 2.0
    assert p._representative_value(pd.Series(["x", "y"]), "median") == "x"

    with pytest.raises(GP3BayesError):
        p._representative_value(
            pd.Series(pd.Categorical([], categories=[])),
            "median",
        )
    with pytest.raises(GP3BayesError):
        p._representative_value(pd.Series([], dtype=bool), "median")
    with pytest.raises(GP3BayesError):
        p._representative_value(pd.Series([np.nan]), "median")
    with pytest.raises(GP3BayesError):
        p._representative_value(pd.Series([None], dtype=object), "median")

    assert p._default_grid_values(categorical, "median") == ["a", "b"]
    assert p._default_grid_values(pd.Series([False, True]), "median") == [False, True]
    assert p._default_grid_values(pd.Series([1.0, 2.0]), "median") == [1.5]
    assert p._default_grid_values(pd.Series(["x", "y", "x"]), "median") == ["x", "y"]

    fit = _fit(3610)
    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(fit, at=[("x", 1)])  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(fit, at={"": 1})
    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(fit, numeric_at="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(fit, variables=("missing",))
    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(fit, at={"missing": 1})
    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(fit, variables=("condition",), at={"condition": []})
    condition_levels = (
        fit.specification.prepared.data["condition"].dropna().drop_duplicates().tolist()
    )
    assert len(condition_levels) == 2
    with pytest.raises(GP3BayesError, match="Prediction grid would contain"):
        p.create_prediction_grid(
            fit,
            variables=("condition", "trial_covariate"),
            at={
                "condition": condition_levels,
                "trial_covariate": list(range(20)),
            },
            max_rows=10,
        )

    grid = p.create_prediction_grid(fit, variables=[])
    assert len(grid) == 1
    required = p._required_prediction_variables(fit)
    assert all(name in grid.columns for name in required)

    grid2 = p.create_prediction_grid(
        fit,
        variables=("condition",),
        at={"condition": condition_levels},
        numeric_at="mean",
    )
    assert len(grid2) == 2


def test_prediction_support_empty_numeric_training_and_repr():
    fit = _fit(3620)
    train = fit.specification.prepared.data.copy()
    train["trial_covariate"] = np.nan
    prepared = replace(fit.specification.prepared, data=train)
    spec = replace(fit.specification, prepared=prepared)
    altered = replace(fit, specification=spec)

    requested = train.head(2).copy()
    audit = p.audit_prediction_support(altered, requested)
    row = audit.table.loc[audit.table["variable"] == "trial_covariate"].iloc[0]
    assert np.isnan(row["training_min"])
    assert row["outside_support"] == 0
    assert "Rows:" in repr(audit)

    with pytest.raises(GP3BayesError):
        p.audit_prediction_support(fit, pd.DataFrame())
    with pytest.raises(GP3BayesError):
        p.audit_prediction_support(fit, object())  # type: ignore[arg-type]
