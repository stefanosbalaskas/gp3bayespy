"""Pre-fit design-support diagnostics.

Source-faithful Python adaptation of gp3bayes 0.5.0
``design-support-diagnostics.R``.  Audits describe limitations but never modify
formulas, remove rows, impute values, or simplify random effects automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .binary import _fixed_model_matrix
from .contracts import ModelContract
from .exceptions import GP3BayesError
from .readiness import audit_model_readiness


def _status(values: list[str] | pd.Series) -> str:
    vals = list(values)
    if "fail" in vals:
        return "fail"
    if "review" in vals:
        return "review"
    return "pass"


def _input(
    x: Any, contract: ModelContract | None = None
) -> tuple[pd.DataFrame, ModelContract, Any | None]:
    specification = None
    if isinstance(x, pd.DataFrame):
        if contract is None:
            raise GP3BayesError("`contract` is required when `x` is a data frame.")
        return x.copy(), contract, None
    if hasattr(x, "specification"):
        specification = x.specification
    elif hasattr(x, "contract") and hasattr(x, "prepared"):
        specification = x
    elif hasattr(x, "contract") and hasattr(x, "data"):
        data = x.data
        if not isinstance(data, pd.DataFrame):
            raise GP3BayesError("Prepared model data must be a pandas DataFrame.")
        return data.copy(), x.contract, None
    if specification is None:
        raise GP3BayesError("`x` must be model data, a prepared object, specification, or fit.")
    prepared = getattr(specification, "prepared", None)
    data = getattr(prepared, "data", None)
    model_contract = getattr(specification, "contract", None)
    if not isinstance(data, pd.DataFrame) or not isinstance(model_contract, ModelContract):
        raise GP3BayesError("The model object does not retain prepared data and a ModelContract.")
    return data.copy(), model_contract, specification


@dataclass(slots=True)
class MissingnessAudit:
    audit_version: str
    status: str
    family: str
    n_rows: int
    declared_columns: tuple[str, ...]
    absent_columns: tuple[str, ...]
    column_table: pd.DataFrame
    grouping_table: pd.DataFrame
    review_fraction: float
    fail_fraction: float
    automatic_exclusion: bool = False
    automatic_imputation: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.column_table.copy()


@dataclass(slots=True)
class FixedEffectDesignAudit:
    audit_version: str
    status: str
    family: str
    formula_text: str
    n_rows: int
    n_columns: int
    rank: int
    full_rank: bool
    condition_number: float
    condition_number_review: float
    condition_number_fail: float
    invariant_columns: tuple[str, ...]
    singular_values: pd.DataFrame
    column_table: pd.DataFrame
    leverage: np.ndarray
    leverage_threshold: float
    high_leverage_rows: np.ndarray
    high_leverage_data_rows: np.ndarray
    model_frame_data_rows: np.ndarray
    error: str | None = None
    automatic_reparameterization: bool = False
    automatic_variable_removal: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.column_table.copy()


@dataclass(slots=True)
class RandomEffectsSupportAudit:
    audit_version: str
    status: str
    family: str
    random_slope_requested: bool
    component_table: pd.DataFrame
    participant_table: pd.DataFrame
    item_table: pd.DataFrame
    slope_table: pd.DataFrame
    error: str | None = None
    automatic_simplification: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.component_table.copy()


@dataclass(slots=True)
class DesignSupportAudit:
    audit_version: str
    status: str
    family: str
    component_table: pd.DataFrame
    readiness: Any
    strict_readiness: Any
    missingness: MissingnessAudit
    fixed_effect_design: FixedEffectDesignAudit
    random_effects_support: RandomEffectsSupportAudit
    separation: Any = None
    automatic_model_change: bool = False
    automatic_exclusion: bool = False

    def to_frame(self) -> pd.DataFrame:
        return self.component_table.copy()


def _declared_columns(contract: ModelContract) -> tuple[str, ...]:
    values: list[str] = []
    for value in contract.mappings.values():
        if isinstance(value, str) and value and value not in values:
            values.append(value)
    for value in contract.predictors:
        if value and value not in values:
            values.append(value)
    interaction = getattr(contract, "interaction", None)
    if interaction:
        for value in interaction:
            if value and value not in values:
                values.append(value)
    return tuple(values)


def audit_missingness_structure(
    x: Any,
    contract: ModelContract | None = None,
    review_fraction: float = 0.05,
    fail_fraction: float = 0.20,
) -> MissingnessAudit:
    if not 0 <= review_fraction <= fail_fraction <= 1:
        raise GP3BayesError(
            "Missingness thresholds must satisfy 0 <= review_fraction <= fail_fraction <= 1."
        )
    data, model_contract, _ = _input(x, contract)
    columns = _declared_columns(model_contract)
    available = [c for c in columns if c in data]
    absent = tuple(c for c in columns if c not in data)
    rows: list[dict[str, Any]] = []
    for column in available:
        n_missing = int(data[column].isna().sum())
        fraction = n_missing / len(data) if len(data) else float("nan")
        if not np.isfinite(fraction) or (n_missing and fraction >= fail_fraction):
            status = "fail"
        elif n_missing and fraction >= review_fraction:
            status = "review"
        else:
            status = "pass"
        rows.append(
            {
                "column": column,
                "n_missing": n_missing,
                "fraction_missing": fraction,
                "status": status,
            }
        )
    column_table = pd.DataFrame(
        rows,
        columns=["column", "n_missing", "fraction_missing", "status"],
    )
    grouping_rows: list[dict[str, Any]] = []
    if available:
        missing_any = ~data.loc[:, available].notna().all(axis=1)
        for group_name in ("participant", "item", "condition"):
            group_column = model_contract.mappings.get(group_name)
            if not isinstance(group_column, str) or group_column not in data:
                continue
            temp = pd.DataFrame({"level": data[group_column], "missing": missing_any})
            for level, frame in temp.dropna(subset=["level"]).groupby(
                "level", observed=False, sort=False
            ):
                grouping_rows.append(
                    {
                        "group": group_name,
                        "level": str(level),
                        "n_rows": len(frame),
                        "n_missing_rows": int(frame["missing"].sum()),
                        "fraction_missing_rows": float(frame["missing"].mean()),
                    }
                )
    grouping_table = pd.DataFrame(
        grouping_rows,
        columns=[
            "group",
            "level",
            "n_rows",
            "n_missing_rows",
            "fraction_missing_rows",
        ],
    )
    overall = (
        "fail"
        if absent
        else _status(column_table["status"].tolist())
        if len(column_table)
        else "pass"
    )
    return MissingnessAudit(
        "0.2",
        overall,
        model_contract.family,
        len(data),
        columns,
        absent,
        column_table,
        grouping_table,
        float(review_fraction),
        float(fail_fraction),
    )


def audit_fixed_effect_design(
    x: Any,
    contract: ModelContract | None = None,
    condition_number_review: float = 30,
    condition_number_fail: float = 100,
    leverage_multiplier: float = 3,
) -> FixedEffectDesignAudit:
    if condition_number_review < 1 or condition_number_fail < condition_number_review:
        raise GP3BayesError("Invalid condition-number thresholds.")
    if leverage_multiplier < 1:
        raise GP3BayesError("`leverage_multiplier` must be at least 1.")
    data, model_contract, specification = _input(x, contract)
    formula_text = (
        getattr(specification, "formula", None)
        or getattr(specification, "formula_text", None)
        or "declared fixed-effects design"
    )
    required = [
        c
        for c in _declared_columns(model_contract)
        if c in data
        and c
        not in {model_contract.mappings.get("participant"), model_contract.mappings.get("item")}
    ]
    complete = data.dropna(subset=required) if required else data.copy()
    if complete.empty:
        return FixedEffectDesignAudit(
            "0.2",
            "fail",
            model_contract.family,
            str(formula_text),
            0,
            0,
            0,
            False,
            float("inf"),
            float(condition_number_review),
            float(condition_number_fail),
            (),
            pd.DataFrame(columns=["index", "singular_value"]),
            pd.DataFrame(columns=["column", "variance", "invariant"]),
            np.array([], dtype=float),
            float("nan"),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
            "No complete rows are available for the declared fixed-effects design.",
        )
    try:
        matrix, names = _fixed_model_matrix(complete, model_contract)
        matrix = np.asarray(matrix, dtype=float)
    except Exception as exc:
        return FixedEffectDesignAudit(
            "0.2",
            "fail",
            model_contract.family,
            str(formula_text),
            len(complete),
            0,
            0,
            False,
            float("inf"),
            float(condition_number_review),
            float(condition_number_fail),
            (),
            pd.DataFrame(columns=["index", "singular_value"]),
            pd.DataFrame(columns=["column", "variance", "invariant"]),
            np.array([], dtype=float),
            float("nan"),
            np.array([], dtype=int),
            np.array([], dtype=int),
            complete.index.to_numpy(dtype=int, copy=False)
            if pd.api.types.is_integer_dtype(complete.index)
            else np.arange(len(complete)),
            str(exc),
        )
    n, p = matrix.shape
    rank = int(np.linalg.matrix_rank(matrix))
    full_rank = rank == p
    singular = np.linalg.svd(matrix, compute_uv=False)
    positive = singular[singular > np.finfo(float).eps]
    condition = (
        float(positive.max() / positive.min())
        if positive.size >= 2
        else (1.0 if positive.size == 1 else float("inf"))
    )
    variances = np.var(matrix, axis=0, ddof=1) if n > 1 else np.full(p, np.nan)
    invariant_mask = ~np.isfinite(variances) | (variances <= np.finfo(float).eps)
    invariant = tuple(
        name
        for name, flag in zip(names, invariant_mask, strict=True)
        if flag and name != "Intercept"
    )
    if rank > 0:
        q, _ = np.linalg.qr(matrix, mode="reduced")
        leverage = np.sum(q[:, : min(rank, q.shape[1])] ** 2, axis=1)
    else:
        leverage = np.zeros(n, dtype=float)
    threshold = float(leverage_multiplier) * max(rank, 1) / n if n else float("nan")
    high = np.flatnonzero(np.isfinite(leverage) & (leverage > threshold))
    if not full_rank or not np.isfinite(condition) or condition >= condition_number_fail:
        status = "fail"
    elif condition >= condition_number_review or invariant or high.size:
        status = "review"
    else:
        status = "pass"
    return FixedEffectDesignAudit(
        "0.2",
        status,
        model_contract.family,
        str(formula_text),
        n,
        p,
        rank,
        full_rank,
        condition,
        float(condition_number_review),
        float(condition_number_fail),
        invariant,
        pd.DataFrame({"index": np.arange(1, singular.size + 1), "singular_value": singular}),
        pd.DataFrame({"column": names, "variance": variances, "invariant": invariant_mask}),
        leverage,
        threshold,
        high + 1,
        complete.index.to_numpy()[high],
        complete.index.to_numpy(),
    )


def audit_random_effects_support(
    x: Any,
    contract: ModelContract | None = None,
    minimum_repeated_rows: int = 2,
    minimum_group_levels: int = 2,
    minimum_condition_cell_rows: int = 2,
) -> RandomEffectsSupportAudit:
    if min(minimum_repeated_rows, minimum_group_levels, minimum_condition_cell_rows) < 1:
        raise GP3BayesError("Random-effects support thresholds must be positive integers.")
    data, model_contract, _ = _input(x, contract)
    participant_col = model_contract.mappings.get("participant")
    item_col = model_contract.mappings.get("item")
    condition_col = model_contract.mappings.get("condition")
    random_slope = bool(getattr(model_contract, "random_slope", False))
    if not isinstance(participant_col, str) or participant_col not in data:
        table = pd.DataFrame(
            {
                "component": ["participant_repetition", "item_crossing", "random_slope_support"],
                "status": ["fail", "not_assessed", "not_assessed"],
            }
        )
        return RandomEffectsSupportAudit(
            "0.2",
            "fail",
            model_contract.family,
            random_slope,
            table,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            f"Participant column is unavailable: {participant_col}",
        )
    counts = data[participant_col].dropna().astype(str).value_counts(sort=False)
    participant_table = pd.DataFrame(
        {
            "participant": counts.index,
            "n_rows": counts.to_numpy(int),
            "sufficient_rows": counts.to_numpy(int) >= minimum_repeated_rows,
        }
    )
    participant_status = (
        "review"
        if len(counts) < minimum_group_levels or (counts < minimum_repeated_rows).any()
        else "pass"
    )
    item_table = pd.DataFrame(columns=["item", "n_participants", "crossed"])
    item_status = "pass"
    if isinstance(item_col, str) and item_col in data:
        crossed = (
            data.dropna(subset=[item_col, participant_col])
            .groupby(item_col, observed=False)[participant_col]
            .nunique()
        )
        item_table = pd.DataFrame(
            {
                "item": crossed.index.astype(str),
                "n_participants": crossed.to_numpy(int),
                "crossed": crossed.to_numpy(int) >= minimum_group_levels,
            }
        )
        if len(crossed) < minimum_group_levels or (~item_table["crossed"]).any():
            item_status = "review"
    slope_table = pd.DataFrame(columns=["participant", "n_condition_levels", "minimum_cell_rows"])
    slope_status = "pass"
    if random_slope:
        if not isinstance(condition_col, str) or condition_col not in data:
            slope_status = "fail"
        else:
            rows: list[dict[str, Any]] = []
            for participant, frame in data.dropna(subset=[participant_col]).groupby(
                participant_col, observed=False
            ):
                cell = frame[condition_col].dropna().value_counts()
                rows.append(
                    {
                        "participant": str(participant),
                        "n_condition_levels": int(cell.size),
                        "minimum_cell_rows": int(cell.min()) if len(cell) else 0,
                    }
                )
            slope_table = pd.DataFrame(rows)
            if (
                slope_table.empty
                or (
                    (slope_table["n_condition_levels"] < 2)
                    | (slope_table["minimum_cell_rows"] < minimum_condition_cell_rows)
                ).any()
            ):
                slope_status = "fail"
    component = pd.DataFrame(
        {
            "component": ["participant_repetition", "item_crossing", "random_slope_support"],
            "status": [participant_status, item_status, slope_status],
        }
    )
    assessed = [s for s in component["status"] if s != "not_assessed"]
    return RandomEffectsSupportAudit(
        "0.2",
        _status(assessed),
        model_contract.family,
        random_slope,
        component,
        participant_table,
        item_table,
        slope_table,
    )


def audit_design_support(
    x: Any,
    contract: ModelContract | None = None,
    separation: bool = False,
    strict_readiness: bool = True,
) -> DesignSupportAudit:
    data, model_contract, _ = _input(x, contract)
    readiness = audit_model_readiness(data, model_contract)
    missingness = audit_missingness_structure(data, model_contract)
    fixed = audit_fixed_effect_design(data, model_contract)
    random = audit_random_effects_support(data, model_contract)
    standard_status = (
        "fail"
        if not readiness.ready
        else ("review" if readiness.status == "ready_with_warnings" else "pass")
    )
    strict_status = standard_status if strict_readiness else "not_assessed"
    separation_result: Any = None
    separation_status = "not_assessed"
    if separation and model_contract.family == "binary":
        # Conservative complete/quasi-separation screen: if any unique fixed-design
        # row maps to only one outcome class, request review.  No variable is dropped.
        outcome = model_contract.mappings.get("outcome")
        try:
            matrix, names = _fixed_model_matrix(data.dropna(), model_contract)
            y = pd.to_numeric(data.loc[data.dropna().index, cast(str, outcome)]).to_numpy(int)
            keys = pd.DataFrame(matrix, columns=names).astype(str).agg("|".join, axis=1)
            classes = pd.DataFrame({"key": keys, "y": y}).groupby("key")["y"].nunique()
            flagged = int((classes == 1).sum())
            separation_status = "review" if flagged else "pass"
            separation_result = {"status": separation_status, "flagged_design_cells": flagged}
        except Exception as exc:
            separation_status = "review"
            separation_result = {"status": "review", "detail": str(exc)}
    component = pd.DataFrame(
        {
            "component": [
                "standard_readiness",
                "strict_readiness",
                "missingness",
                "fixed_effect_design",
                "random_effects_support",
                "separation",
            ],
            "status": [
                standard_status,
                strict_status,
                missingness.status,
                fixed.status,
                random.status,
                separation_status,
            ],
        }
    )
    assessed = component.loc[component["status"] != "not_assessed", "status"].tolist()
    return DesignSupportAudit(
        "0.2",
        _status(assessed),
        model_contract.family,
        component,
        readiness,
        readiness if strict_readiness else None,
        missingness,
        fixed,
        random,
        separation_result,
    )


def preflight_model_specification(
    specification: Any,
    contract: ModelContract | None = None,
    separation: bool = False,
    strict_readiness: bool = True,
) -> DesignSupportAudit:
    return audit_design_support(
        specification, contract=contract, separation=separation, strict_readiness=strict_readiness
    )


__all__ = [
    "DesignSupportAudit",
    "FixedEffectDesignAudit",
    "MissingnessAudit",
    "RandomEffectsSupportAudit",
    "audit_design_support",
    "audit_fixed_effect_design",
    "audit_missingness_structure",
    "audit_random_effects_support",
    "preflight_model_specification",
]
