"""Shared conservative posterior-predictive check infrastructure."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError
from .posterior import _worst_status

_ArrayLikeFloat = Sequence[float] | np.ndarray[Any, Any]
_ArrayLikeObject = Sequence[object] | np.ndarray[Any, Any]


def _integer(value: object, name: str, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) < minimum
        or float(value) != math.floor(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be an integer greater than or equal to {minimum}.")
    return int(float(value))


def _probability(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or not 0 < float(value) < 1
    ):
        raise GP3BayesError(f"`{name}` must be one finite probability strictly between 0 and 1.")
    return float(value)


def _validate_controls(
    draws: object,
    seed: object,
    pass_probability: object,
    review_probability: object,
) -> tuple[int, int, float, float]:
    draw_count = _integer(draws, "draws", minimum=50)
    seed_value = _integer(seed, "seed", minimum=0)
    pass_value = _probability(pass_probability, "pass_probability")
    review_value = _probability(review_probability, "review_probability")
    if pass_value >= review_value:
        raise GP3BayesError(
            "`pass_probability` must be smaller than `review_probability`."
        )
    return draw_count, seed_value, pass_value, review_value


def _sample_sd(values: _ArrayLikeFloat) -> float:
    array = np.asarray(values, dtype=float)
    return math.nan if array.size <= 1 else float(np.std(array, ddof=1))


def _group_means(values: np.ndarray, groups: _ArrayLikeObject) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": list(groups)})
    return frame.groupby("group", sort=False, observed=False)["value"].mean().to_numpy(dtype=float)


def _group_medians(values: np.ndarray, groups: _ArrayLikeObject) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": list(groups)})
    grouped = frame.groupby("group", sort=False, observed=False)["value"].median()
    return grouped.to_numpy(dtype=float)


def _binary_summary(
    y: _ArrayLikeFloat,
    *,
    condition: _ArrayLikeObject | None,
    participant: _ArrayLikeObject,
    item: _ArrayLikeObject | None,
) -> dict[str, float]:
    values = np.asarray(y, dtype=float)
    participant_rates = _group_means(values, participant)
    low_rate = math.nan
    high_rate = math.nan
    contrast = math.nan

    if condition is not None:
        condition_values = np.asarray(condition)
        levels = sorted(pd.unique(condition_values).tolist())
        if len(levels) == 2:
            rates = [float(np.mean(values[condition_values == level])) for level in levels]
            low_rate, high_rate = rates
            contrast = high_rate - low_rate

    item_rate_sd = math.nan
    if item is not None:
        item_rates = _group_means(values, item)
        item_rate_sd = _sample_sd(item_rates)

    return {
        "overall_rate": float(np.mean(values)),
        "condition_low_rate": low_rate,
        "condition_high_rate": high_rate,
        "condition_rate_contrast": contrast,
        "participant_rate_sd": _sample_sd(participant_rates),
        "item_rate_sd": item_rate_sd,
    }


def _duration_summary(
    y: _ArrayLikeFloat,
    *,
    condition: _ArrayLikeObject | None,
    participant: _ArrayLikeObject,
    item: _ArrayLikeObject | None,
) -> dict[str, float]:
    values = np.asarray(y, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    nonfinite_fraction = float(np.mean(~valid))
    if not np.any(valid):
        return {
            "median": math.inf,
            "mean": math.inf,
            "q90": math.inf,
            "q99": math.inf,
            "coefficient_of_variation": math.inf,
            "condition_median_ratio": math.nan,
            "participant_log_median_sd": math.nan,
            "item_log_median_sd": math.nan,
            "nonfinite_fraction": nonfinite_fraction,
        }

    y_valid = values[valid]
    participant_values = np.asarray(participant, dtype=object)[valid]
    condition_values = None if condition is None else np.asarray(condition, dtype=object)[valid]
    item_values = None if item is None else np.asarray(item, dtype=object)[valid]
    log_y = np.log(y_valid)
    participant_medians = _group_medians(log_y, participant_values)

    condition_ratio = math.nan
    if condition_values is not None:
        levels = sorted(pd.unique(condition_values).tolist())
        if len(levels) == 2:
            medians = [
                float(np.median(y_valid[condition_values == level])) for level in levels
            ]
            if all(math.isfinite(value) for value in medians) and medians[0] > 0:
                condition_ratio = medians[1] / medians[0]

    item_log_median_sd = math.nan
    if item_values is not None:
        item_medians = _group_medians(log_y, item_values)
        item_log_median_sd = _sample_sd(item_medians)

    mean_y = float(np.mean(y_valid))
    cv = (
        float(np.std(y_valid, ddof=1) / mean_y)
        if y_valid.size > 1 and math.isfinite(mean_y) and mean_y > 0
        else math.nan
    )
    return {
        "median": float(np.median(y_valid)),
        "mean": mean_y,
        "q90": float(np.quantile(y_valid, 0.90, method="median_unbiased")),
        "q99": float(np.quantile(y_valid, 0.99, method="median_unbiased")),
        "coefficient_of_variation": cv,
        "condition_median_ratio": condition_ratio,
        "participant_log_median_sd": _sample_sd(participant_medians),
        "item_log_median_sd": item_log_median_sd,
        "nonfinite_fraction": nonfinite_fraction,
    }


def _predictive_status(
    observed: float,
    pass_lower: float,
    pass_upper: float,
    review_lower: float,
    review_upper: float,
) -> str:
    if not all(
        math.isfinite(value)
        for value in (observed, pass_lower, pass_upper, review_lower, review_upper)
    ):
        return "not_applicable"
    if pass_lower <= observed <= pass_upper:
        return "pass"
    if review_lower <= observed <= review_upper:
        return "review"
    return "fail"


def _replicated_table(
    yrep: np.ndarray,
    *,
    summary_function,
    condition: _ArrayLikeObject | None,
    participant: _ArrayLikeObject,
    item: _ArrayLikeObject | None,
) -> pd.DataFrame:
    rows = [
        summary_function(
            row,
            condition=condition,
            participant=participant,
            item=item,
        )
        for row in yrep
    ]
    return pd.DataFrame(rows)


def _check_table(
    observed: Mapping[str, float],
    replicated: pd.DataFrame,
    *,
    pass_probability: float,
    review_probability: float,
) -> pd.DataFrame:
    pass_alpha = (1 - pass_probability) / 2
    review_alpha = (1 - review_probability) / 2
    rows: list[dict[str, float | str]] = []
    for statistic, observed_value in observed.items():
        values = pd.to_numeric(replicated[statistic], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0 or not math.isfinite(float(observed_value)):
            rows.append(
                {
                    "statistic": statistic,
                    "observed": float(observed_value),
                    "replicated_median": math.nan,
                    "pass_lower": math.nan,
                    "pass_upper": math.nan,
                    "review_lower": math.nan,
                    "review_upper": math.nan,
                    "status": "not_applicable",
                }
            )
            continue
        pass_interval = np.quantile(
            values,
            [pass_alpha, 1 - pass_alpha],
            method="median_unbiased",
        )
        review_interval = np.quantile(
            values,
            [review_alpha, 1 - review_alpha],
            method="median_unbiased",
        )
        rows.append(
            {
                "statistic": statistic,
                "observed": float(observed_value),
                "replicated_median": float(np.median(values)),
                "pass_lower": float(pass_interval[0]),
                "pass_upper": float(pass_interval[1]),
                "review_lower": float(review_interval[0]),
                "review_upper": float(review_interval[1]),
                "status": _predictive_status(
                    float(observed_value),
                    float(pass_interval[0]),
                    float(pass_interval[1]),
                    float(review_interval[0]),
                    float(review_interval[1]),
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class _BinaryPosteriorPredictiveCheck:
    check_version: str
    family: str
    draws: int
    seed: int
    observed: Mapping[str, float]
    replicated: pd.DataFrame
    checks: pd.DataFrame
    brier_score: float
    status: str
    posterior_predictive_performed: bool = True
    adequacy_established: bool = False
    interpretation: str = (
        "The status describes selected posterior predictive summaries and is not a "
        "global declaration of model adequacy."
    )


@dataclass(frozen=True, slots=True)
class _DurationPosteriorPredictiveCheck:
    check_version: str
    family: str
    outcome_unit: str
    draws: int
    seed: int
    observed: Mapping[str, float]
    replicated: pd.DataFrame
    checks: pd.DataFrame
    log_scale_rmse: float
    status: str
    posterior_predictive_performed: bool = True
    adequacy_established: bool = False
    interpretation: str = (
        "The status describes selected duration predictive summaries and is not a "
        "global declaration of model adequacy."
    )


def _binary_result(
    *,
    draws: int,
    seed: int,
    observed: Mapping[str, float],
    replicated: pd.DataFrame,
    checks: pd.DataFrame,
    brier_score: float,
) -> _BinaryPosteriorPredictiveCheck:
    return _BinaryPosteriorPredictiveCheck(
        check_version="0.1",
        family="binary",
        draws=draws,
        seed=seed,
        observed=observed,
        replicated=replicated,
        checks=checks,
        brier_score=brier_score,
        status=_worst_status(checks["status"].astype(str).tolist()),
    )


def _duration_result(
    *,
    outcome_unit: str,
    draws: int,
    seed: int,
    observed: Mapping[str, float],
    replicated: pd.DataFrame,
    checks: pd.DataFrame,
    log_scale_rmse: float,
) -> _DurationPosteriorPredictiveCheck:
    return _DurationPosteriorPredictiveCheck(
        check_version="0.1",
        family="duration",
        outcome_unit=outcome_unit,
        draws=draws,
        seed=seed,
        observed=observed,
        replicated=replicated,
        checks=checks,
        log_scale_rmse=log_scale_rmse,
        status=_worst_status(checks["status"].astype(str).tolist()),
    )
