"""Governed posterior prediction for approved gp3bayespy fits.

This module ports the first prediction-support tranche from frozen gp3bayes
0.5.0. Predictions are descriptive posterior quantities under the fitted
model. Support flags never remove rows automatically, and prediction does not
establish causal effects or out-of-sample adequacy.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy.special import expit  # type: ignore[import-untyped]

from .binary import BinaryFit, BinaryModelSpecification, _fixed_model_matrix
from .duration import DurationFit, DurationModelSpecification
from .exceptions import GP3BayesError

_PredictionType = Literal["expected", "predictive", "linear", "median"]
_NumericAt = Literal["median", "mean"]

_Fit = BinaryFit | DurationFit
_Specification = BinaryModelSpecification | DurationModelSpecification


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise GP3BayesError(f"`{name}` must be TRUE or FALSE.")
    return value


def _positive_integer_or_none(value: object, name: str) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) < 1
        or float(value) != math.floor(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be NULL or one positive integer.")
    return int(float(value))


def _nonnegative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) < 0
        or float(value) != math.floor(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be one non-negative integer.")
    return int(float(value))


def _positive_integer(value: object, name: str) -> int:
    result = _positive_integer_or_none(value, name)
    if result is None:
        raise GP3BayesError(f"`{name}` must be one positive integer.")
    return result


def _probabilities(values: Sequence[float]) -> tuple[float, float, float]:
    try:
        probs = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError("`probs` must contain three finite probabilities.") from exc
    if (
        len(probs) != 3
        or any(not math.isfinite(value) for value in probs)
        or any(value < 0 or value > 1 for value in probs)
        or not probs[0] < probs[1] < probs[2]
    ):
        raise GP3BayesError(
            "`probs` must contain three strictly increasing probabilities in [0, 1]."
        )
    return cast(tuple[float, float, float], probs)


def _validate_fit(fit: object, family: str | None = None) -> _Fit:
    if not isinstance(fit, (BinaryFit, DurationFit)):
        raise GP3BayesError("`fit` must inherit from `gp3bayes_fit`.")
    if family is not None and fit.family != family:
        raise GP3BayesError(f"`fit` must use the approved {family} family.")
    posterior = getattr(fit.backend_fit, "posterior", None)
    if posterior is None:
        raise GP3BayesError("`fit.backend_fit` must contain posterior draws.")
    return fit


def _object_parts(
    x: object,
) -> tuple[_Specification, pd.DataFrame, Any, str]:
    if isinstance(x, (BinaryFit, DurationFit)):
        specification: _Specification = x.specification
    elif isinstance(x, (BinaryModelSpecification, DurationModelSpecification)):
        specification = x
    else:
        raise GP3BayesError("`x` must be a gp3bayes fit or model specification.")

    prepared = specification.prepared
    if prepared is None:
        raise GP3BayesError("The model specification must retain prepared model data.")
    data = prepared.data
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise GP3BayesError("Prepared model data must be a non-empty data frame.")
    return specification, data, specification.contract, specification.family


def _as_values(value: object) -> list[Any]:
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.Index):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim else [value.item()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _restore_type(values: Sequence[Any], template: pd.Series) -> pd.Series:
    data = list(values)
    if isinstance(template.dtype, pd.CategoricalDtype):
        return pd.Series(
            pd.Categorical(
                data,
                categories=template.cat.categories,
                ordered=template.cat.ordered,
            )
        )
    if pd.api.types.is_integer_dtype(template.dtype):
        return pd.Series(data, dtype=template.dtype)
    if pd.api.types.is_bool_dtype(template.dtype):
        return pd.Series(data, dtype=bool)
    if pd.api.types.is_string_dtype(template.dtype) or pd.api.types.is_object_dtype(
        template.dtype
    ):
        return pd.Series(data, dtype=object)
    if pd.api.types.is_numeric_dtype(template.dtype):
        return pd.Series(pd.to_numeric(pd.Series(data), errors="raise"), dtype=float)
    return pd.Series(data)


def _representative_value(template: pd.Series, numeric_at: _NumericAt) -> Any:
    if isinstance(template.dtype, pd.CategoricalDtype):
        categories = list(template.cat.categories)
        if not categories:
            raise GP3BayesError("No prediction values are available for a categorical variable.")
        return categories[0]
    if pd.api.types.is_bool_dtype(template.dtype):
        values = list(pd.unique(template.dropna()))
        if not values:
            raise GP3BayesError("No prediction values are available for a logical variable.")
        return values[0]
    if pd.api.types.is_numeric_dtype(template.dtype):
        numeric = pd.to_numeric(template, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).any():
            raise GP3BayesError("No finite prediction values are available for a numeric variable.")
        if numeric_at == "median":
            return float(np.nanmedian(numeric))
        return float(np.nanmean(numeric))
    values = list(pd.unique(template.dropna()))
    if not values:
        raise GP3BayesError("No prediction values are available for a predictor.")
    return values[0]


def _default_grid_values(template: pd.Series, numeric_at: _NumericAt) -> list[Any]:
    if isinstance(template.dtype, pd.CategoricalDtype):
        return list(template.cat.categories)
    if pd.api.types.is_bool_dtype(template.dtype):
        return list(pd.unique(template.dropna()))
    if pd.api.types.is_numeric_dtype(template.dtype):
        return [_representative_value(template, numeric_at)]
    return list(pd.unique(template.dropna()))


def create_prediction_grid(
    x: object,
    variables: Sequence[str] | str | None = None,
    at: Mapping[str, object] | None = None,
    numeric_at: _NumericAt = "median",
    max_rows: int = 5000,
) -> pd.DataFrame:
    """Create a governed Cartesian prediction grid from declared predictors."""
    _, data, contract, _ = _object_parts(x)
    if at is None:
        at_map: Mapping[str, object] = {}
    elif not isinstance(at, Mapping) or any(not isinstance(key, str) or not key for key in at):
        raise GP3BayesError("`at` must be a named mapping.")
    else:
        at_map = at
    if numeric_at not in {"median", "mean"}:
        raise GP3BayesError('`numeric_at` must be either "median" or "mean".')
    max_rows_value = _positive_integer(max_rows, "max_rows")

    declared: list[str] = []
    for value in (
        contract.mappings.get("condition"),
        contract.mappings.get("time"),
        *contract.predictors,
    ):
        if value is not None and value in data.columns and value not in declared:
            declared.append(value)

    if variables is None:
        selected = declared
    else:
        selected = [variables] if isinstance(variables, str) else list(variables)
        if any(not isinstance(value, str) or value not in data.columns for value in selected):
            raise GP3BayesError(
                "`variables` must identify columns in the prepared model data."
            )
        selected = list(dict.fromkeys(selected))

    unknown_at = [name for name in at_map if name not in data.columns]
    if unknown_at:
        raise GP3BayesError("Unknown `at` variables: " + ", ".join(unknown_at) + ".")

    value_map: dict[str, list[Any]] = {}
    for name in selected:
        template = data[name]
        values = (
            _as_values(at_map[name])
            if name in at_map
            else _default_grid_values(template, numeric_at)
        )
        if not values:
            raise GP3BayesError(f"No prediction values available for `{name}`.")
        # Validate explicit values against the prepared column type.
        restored = _restore_type(values, template)
        value_map[name] = restored.tolist()

    grid_n = math.prod(len(values) for values in value_map.values()) if value_map else 1
    if grid_n > max_rows_value:
        raise GP3BayesError(
            f"Prediction grid would contain {grid_n} rows; reduce `variables`/`at` "
            "or increase `max_rows` explicitly."
        )

    if value_map:
        import itertools

        rows = [
            dict(zip(value_map, combination, strict=True))
            for combination in itertools.product(*(value_map[name] for name in value_map))
        ]
        grid = pd.DataFrame(rows)
        for name in value_map:
            grid[name] = _restore_type(grid[name].tolist(), data[name]).to_numpy()
    else:
        grid = pd.DataFrame(index=pd.RangeIndex(1))

    # Add fixed and group variables needed by the approved model using observed,
    # deterministic representative values. This mirrors the R grid's completion
    # step and never invents a new factor/group level.
    required: list[str] = []
    for value in (
        contract.mappings.get("condition"),
        contract.mappings.get("time"),
        *contract.predictors,
        contract.mappings.get("participant"),
        contract.mappings.get("item"),
    ):
        if value is not None and value in data.columns and value not in required:
            required.append(value)

    for name in required:
        if name in grid.columns:
            continue
        template = data[name]
        if name in at_map:
            explicit = _as_values(at_map[name])
            if not explicit:
                raise GP3BayesError(f"No prediction values available for `{name}`.")
            value = explicit[0]
        else:
            value = _representative_value(template, numeric_at)
        grid[name] = _restore_type([value] * len(grid), template).to_numpy()

    return grid.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class PredictionSupport:
    """Descriptive support audit for requested prediction rows."""

    table: pd.DataFrame
    rows: int
    has_extrapolation: bool
    has_novel_levels: bool
    has_missing_required: bool
    automatic_rejection: bool = False
    interpretation: str = (
        "Support flags identify extrapolation or novel factor levels. "
        "Rows are not excluded automatically."
    )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_prediction_support>",
                f"  Rows: {self.rows}",
                f"  Extrapolation: {str(self.has_extrapolation).upper()}",
                f"  Novel levels: {str(self.has_novel_levels).upper()}",
                f"  Missing required variables: {str(self.has_missing_required).upper()}",
                "  Automatic rejection: FALSE",
            ]
        )


def _required_prediction_variables(fit: _Fit) -> list[str]:
    contract = fit.specification.contract
    required: list[str] = []
    for value in (
        contract.mappings.get("condition"),
        contract.mappings.get("time"),
        *contract.predictors,
        contract.mappings.get("participant"),
        contract.mappings.get("item"),
    ):
        if value is not None and value not in required:
            required.append(value)
    return required


def audit_prediction_support(fit: _Fit, newdata: pd.DataFrame) -> PredictionSupport:
    """Compare requested prediction rows with observed model-building support."""
    validated = _validate_fit(fit)
    if not isinstance(newdata, pd.DataFrame) or newdata.empty:
        raise GP3BayesError("`newdata` must be a non-empty data frame.")
    train = cast(Any, validated.specification.prepared).data
    rows: list[dict[str, Any]] = []

    for name in _required_prediction_variables(validated):
        if name not in newdata.columns:
            rows.append(
                {
                    "variable": name,
                    "type": "missing",
                    "training_min": math.nan,
                    "training_max": math.nan,
                    "outside_support": len(newdata),
                    "novel_levels": math.nan,
                    "missing_values": math.nan,
                    "detail": "required variable absent from newdata",
                }
            )
            continue

        training = train[name]
        requested = newdata[name]
        if pd.api.types.is_numeric_dtype(training.dtype):
            numeric_train = pd.to_numeric(training, errors="coerce").to_numpy(dtype=float)
            finite_train = numeric_train[np.isfinite(numeric_train)]
            if finite_train.size == 0:
                lo = hi = math.nan
                outside = 0
            else:
                lo = float(np.min(finite_train))
                hi = float(np.max(finite_train))
                numeric_new = pd.to_numeric(requested, errors="coerce").to_numpy(dtype=float)
                outside = int(
                    np.sum(np.isfinite(numeric_new) & ((numeric_new < lo) | (numeric_new > hi)))
                )
            rows.append(
                {
                    "variable": name,
                    "type": "numeric",
                    "training_min": lo,
                    "training_max": hi,
                    "outside_support": outside,
                    "novel_levels": math.nan,
                    "missing_values": int(requested.isna().sum()),
                    "detail": (
                        "values extend beyond observed range"
                        if outside
                        else "within observed range"
                    ),
                }
            )
        else:
            training_levels = set(training.dropna().astype(str).tolist())
            requested_levels = set(requested.dropna().astype(str).tolist())
            novel = sorted(requested_levels - training_levels)
            rows.append(
                {
                    "variable": name,
                    "type": "categorical",
                    "training_min": math.nan,
                    "training_max": math.nan,
                    "outside_support": 0,
                    "novel_levels": len(novel),
                    "missing_values": int(requested.isna().sum()),
                    "detail": (
                        "novel: " + ", ".join(novel)
                        if novel
                        else "all levels observed in training data"
                    ),
                }
            )

    table = pd.DataFrame(
        rows,
        columns=[
            "variable",
            "type",
            "training_min",
            "training_max",
            "outside_support",
            "novel_levels",
            "missing_values",
            "detail",
        ],
    )
    return PredictionSupport(
        table=table,
        rows=len(newdata),
        has_extrapolation=bool(
            len(table) and (table["outside_support"].fillna(0).astype(float) > 0).any()
        ),
        has_novel_levels=bool(
            len(table) and (table["novel_levels"].fillna(0).astype(float) > 0).any()
        ),
        has_missing_required=bool(len(table) and table["type"].eq("missing").any()),
    )


def prediction_support_table(x: PredictionSupport) -> pd.DataFrame:
    """Return the underlying prediction-support audit table."""
    if not isinstance(x, PredictionSupport):
        raise GP3BayesError("`x` must be a gp3bayes prediction-support audit.")
    return x.table.copy()


def _posterior_values(fit: _Fit, name: str) -> np.ndarray:
    posterior = cast(Any, fit.backend_fit).posterior
    if name not in posterior:
        raise GP3BayesError(f"Posterior variable `{name}` is unavailable.")
    values = np.asarray(posterior[name].values, dtype=float)
    if values.ndim < 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise GP3BayesError(f"Posterior variable `{name}` has an invalid draw shape.")
    if not np.isfinite(values).all():
        raise GP3BayesError(f"Posterior variable `{name}` contains non-finite draws.")
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def _draw_count(fit: _Fit) -> int:
    return int(_posterior_values(fit, "b_Intercept").shape[0])


def _take_draws(values: np.ndarray, n: int) -> np.ndarray:
    return values[:n]


def _fixed_matrix_for_prediction(
    fit: _Fit, newdata: pd.DataFrame
) -> tuple[np.ndarray, tuple[str, ...]]:
    prepared = cast(Any, fit.specification.prepared)
    training = prepared.data
    contract = fit.specification.contract
    fixed_names: list[str] = []
    for value in (
        contract.mappings.get("condition"),
        contract.mappings.get("time"),
        *contract.predictors,
    ):
        if value is not None and value not in fixed_names:
            fixed_names.append(value)

    missing = [name for name in fixed_names if name not in newdata.columns]
    if missing:
        raise GP3BayesError(
            "Prediction data are missing fixed-effect variables: "
            + ", ".join(missing)
            + "."
        )
    combined = pd.concat(
        [training.loc[:, fixed_names], newdata.loc[:, fixed_names]],
        ignore_index=True,
    )
    matrix, names = _fixed_model_matrix(combined, contract)
    expected_names = tuple(prepared.model_matrix_columns)
    if names != expected_names:
        raise GP3BayesError(
            "Prediction data introduce an unsupported fixed-effect level or alter "
            "the approved design matrix."
        )
    return matrix[-len(newdata) :, :], names


def _group_effect_matrix(
    fit: _Fit,
    newdata: pd.DataFrame,
    *,
    ndraws: int,
    allow_new_levels: bool,
    seed: int,
) -> np.ndarray:
    contract = fit.specification.contract
    prepared = cast(Any, fit.specification.prepared)
    training = prepared.data
    result = np.zeros((ndraws, len(newdata)), dtype=float)
    rng = np.random.default_rng(seed)

    participant_col = contract.mappings.get("participant")
    if participant_col is not None:
        if participant_col not in newdata.columns:
            raise GP3BayesError(
                f"Prediction data must contain grouping variable `{participant_col}` "
                "when `include_group_effects` is TRUE."
            )
        participant_levels = list(dict.fromkeys(training[participant_col].tolist()))
        lookup = {str(value): index for index, value in enumerate(participant_levels)}
        requested = newdata[participant_col].astype(str).tolist()
        missing_levels = sorted(set(requested) - set(lookup))
        if missing_levels and not allow_new_levels:
            raise GP3BayesError(
                "New participant levels require `allow_new_levels = TRUE`: "
                + ", ".join(missing_levels)
                + "."
            )

        if contract.random_slope:
            participant_re = _take_draws(_posterior_values(fit, "participant_re"), ndraws)
            if participant_re.ndim != 3 or participant_re.shape[2] != 2:
                raise GP3BayesError("Participant random-slope draws have an invalid shape.")
            condition_col = contract.mappings.get("condition")
            if condition_col is None or condition_col not in newdata.columns:
                raise GP3BayesError(
                    "Prediction data must contain the condition for participant random slopes."
                )
            condition = pd.to_numeric(
                newdata[condition_col], errors="raise"
            ).to_numpy(dtype=float)
            participant_slope_cache: dict[str, np.ndarray] = {}
            for row, level in enumerate(requested):
                if level in lookup:
                    effect = participant_re[:, lookup[level], :]
                else:
                    if level not in participant_slope_cache:
                        sampled = rng.integers(
                            0, participant_re.shape[1], size=ndraws
                        )
                        participant_slope_cache[level] = participant_re[
                            np.arange(ndraws), sampled, :
                        ]
                    effect = participant_slope_cache[level]
                result[:, row] += effect[:, 0] + effect[:, 1] * condition[row]
        else:
            sd = _take_draws(_posterior_values(fit, "sd_participant"), ndraws).reshape(ndraws)
            z = _take_draws(_posterior_values(fit, "participant_z"), ndraws)
            if z.ndim != 2:
                raise GP3BayesError("Participant random-intercept draws have an invalid shape.")
            known_effect = sd[:, None] * z
            participant_intercept_cache: dict[str, np.ndarray] = {}
            for row, level in enumerate(requested):
                if level in lookup:
                    result[:, row] += known_effect[:, lookup[level]]
                else:
                    if level not in participant_intercept_cache:
                        sampled = rng.integers(0, known_effect.shape[1], size=ndraws)
                        participant_intercept_cache[level] = known_effect[
                            np.arange(ndraws), sampled
                        ]
                    result[:, row] += participant_intercept_cache[level]

    item_col = contract.mappings.get("item")
    if item_col is not None:
        if item_col not in newdata.columns:
            raise GP3BayesError(
                f"Prediction data must contain grouping variable `{item_col}` "
                "when `include_group_effects` is TRUE."
            )
        item_levels = list(dict.fromkeys(training[item_col].tolist()))
        lookup = {str(value): index for index, value in enumerate(item_levels)}
        requested = newdata[item_col].astype(str).tolist()
        missing_levels = sorted(set(requested) - set(lookup))
        if missing_levels and not allow_new_levels:
            raise GP3BayesError(
                "New item levels require `allow_new_levels = TRUE`: "
                + ", ".join(missing_levels)
                + "."
            )
        sd = _take_draws(_posterior_values(fit, "sd_item"), ndraws).reshape(ndraws)
        z = _take_draws(_posterior_values(fit, "item_z"), ndraws)
        if z.ndim != 2:
            raise GP3BayesError("Item random-intercept draws have an invalid shape.")
        known_effect = sd[:, None] * z
        item_cache: dict[str, np.ndarray] = {}
        for row, level in enumerate(requested):
            if level in lookup:
                result[:, row] += known_effect[:, lookup[level]]
            else:
                if level not in item_cache:
                    sampled = rng.integers(0, known_effect.shape[1], size=ndraws)
                    item_cache[level] = known_effect[np.arange(ndraws), sampled]
                result[:, row] += item_cache[level]

    return result


def _linear_prediction_matrix(
    fit: _Fit,
    newdata: pd.DataFrame,
    *,
    include_group_effects: bool,
    allow_new_levels: bool,
    ndraws: int | None,
    seed: int,
) -> np.ndarray:
    matrix, _ = _fixed_matrix_for_prediction(fit, newdata)
    total = _draw_count(fit)
    draw_n = total if ndraws is None else ndraws
    if draw_n > total:
        raise GP3BayesError(
            f"`ndraws` ({draw_n}) exceeds the {total} posterior draws available."
        )

    intercept = _take_draws(_posterior_values(fit, "b_Intercept"), draw_n).reshape(draw_n)
    eta = np.repeat(intercept[:, None], len(newdata), axis=1)
    if matrix.shape[1] > 1:
        beta = _take_draws(_posterior_values(fit, "b"), draw_n)
        if beta.ndim == 1:
            beta = beta[:, None]
        if beta.shape[1] != matrix.shape[1] - 1:
            raise GP3BayesError(
                "Posterior fixed-effect dimensions do not match the prediction design."
            )
        eta += beta @ matrix[:, 1:].T

    if include_group_effects:
        eta += _group_effect_matrix(
            fit,
            newdata,
            ndraws=draw_n,
            allow_new_levels=allow_new_levels,
            seed=seed,
        )
    return eta


def _prediction_summary(
    draws: np.ndarray,
    probs: tuple[float, float, float],
    observed: pd.Series | None,
) -> pd.DataFrame:
    quantiles = np.quantile(draws, probs, axis=0, method="linear")
    ddof = 1 if draws.shape[0] > 1 else 0
    summary = pd.DataFrame(
        {
            "observation": np.arange(1, draws.shape[1] + 1, dtype=int),
            "predicted_mean": np.mean(draws, axis=0),
            "predicted_sd": np.std(draws, axis=0, ddof=ddof),
            "lower": quantiles[0],
            "predicted_median": quantiles[1],
            "upper": quantiles[2],
        }
    )
    if observed is not None:
        summary["observed"] = observed.to_numpy(copy=True)
    return summary


@dataclass(frozen=True, slots=True)
class Prediction:
    """Posterior prediction draws, summaries, support audit, and provenance."""

    family: str
    type: _PredictionType
    scale: str
    draws: np.ndarray
    summary: pd.DataFrame
    newdata: pd.DataFrame
    observed: pd.Series | None
    support: PredictionSupport
    include_group_effects: bool
    allow_new_levels: bool
    probs: tuple[float, float, float]
    seed: int
    automatic_decision: bool = False
    causal_effect_established: bool = False
    out_of_sample_adequacy_established: bool = False
    interpretation: str = (
        "Predictions are descriptive posterior quantities under the fitted model. "
        "They do not establish causal effects or out-of-sample adequacy."
    )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_prediction>",
                f"  Family: {self.family}",
                f"  Type: {self.type}",
                f"  Rows: {len(self.summary)}",
                f"  Draws: {self.draws.shape[0]}",
                "  Include group effects: "
                + str(self.include_group_effects).upper(),
                "  Automatic decision: FALSE",
            ]
        )


def predict_model(
    fit: _Fit,
    newdata: pd.DataFrame | None = None,
    type: _PredictionType = "expected",
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    seed: int = 1,
) -> Prediction:
    """Generate governed posterior predictions for an approved fitted model."""
    validated = _validate_fit(fit)
    if type not in {"expected", "predictive", "linear", "median"}:
        raise GP3BayesError(
            '`type` must be one of "expected", "predictive", "linear", or "median".'
        )
    include_re = _flag(include_group_effects, "include_group_effects")
    allow_new = _flag(allow_new_levels, "allow_new_levels")
    draw_n = _positive_integer_or_none(ndraws, "ndraws")
    seed_value = _nonnegative_integer(seed, "seed")
    probs_value = _probabilities(probs)
    if type == "median" and validated.family != "duration":
        raise GP3BayesError('`type = "median"` is available only for duration models.')

    prepared = cast(Any, validated.specification.prepared)
    prediction_data = prepared.data.copy() if newdata is None else newdata.copy()
    if not isinstance(prediction_data, pd.DataFrame) or prediction_data.empty:
        raise GP3BayesError("Prediction data must be a non-empty data frame.")
    support = audit_prediction_support(validated, prediction_data)

    outcome_col = cast(str, validated.specification.contract.mappings["outcome"])
    observed = (
        prediction_data[outcome_col].copy()
        if outcome_col in prediction_data.columns
        else None
    )
    eta = _linear_prediction_matrix(
        validated,
        prediction_data,
        include_group_effects=include_re,
        allow_new_levels=allow_new,
        ndraws=draw_n,
        seed=seed_value,
    )

    if type == "linear":
        draws = eta
        scale = "log_odds" if validated.family == "binary" else "log_duration_location"
    elif validated.family == "binary":
        probability = expit(eta)
        if type == "predictive":
            rng = np.random.default_rng(seed_value)
            draws = rng.binomial(1, probability).astype(float)
        else:
            draws = np.asarray(probability, dtype=float)
        scale = "response"
    else:
        if type == "median":
            draws = np.exp(eta)
            scale = "duration_median"
        else:
            sigma = _take_draws(
                _posterior_values(validated, "sigma"), eta.shape[0]
            ).reshape(eta.shape[0], 1)
            if type == "expected":
                draws = np.exp(eta + 0.5 * sigma**2)
            else:
                rng = np.random.default_rng(seed_value)
                draws = rng.lognormal(mean=eta, sigma=sigma)
            scale = "response"

    if draws.size == 0 or not np.isfinite(draws).all():
        raise GP3BayesError("Posterior predictions were not returned as a finite matrix.")
    summary = _prediction_summary(draws, probs_value, observed)

    return Prediction(
        family=validated.family,
        type=type,
        scale=scale,
        draws=np.asarray(draws, dtype=float),
        summary=summary,
        newdata=prediction_data,
        observed=observed,
        support=support,
        include_group_effects=include_re,
        allow_new_levels=allow_new,
        probs=probs_value,
        seed=seed_value,
    )


def prediction_table(x: Prediction) -> pd.DataFrame:
    """Return the observation-level posterior prediction summary."""
    if not isinstance(x, Prediction):
        raise GP3BayesError("`x` must be a gp3bayes prediction.")
    return x.summary.copy()


def extract_expected_predictions(
    fit: _Fit,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
) -> np.ndarray:
    """Extract conditional expected-response posterior draws."""
    return predict_model(
        fit,
        newdata=newdata,
        type="expected",
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
    ).draws


def extract_posterior_predictions(
    fit: _Fit,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
    seed: int = 1,
) -> np.ndarray:
    """Extract new-outcome posterior predictive draws."""
    return predict_model(
        fit,
        newdata=newdata,
        type="predictive",
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
        seed=seed,
    ).draws


def extract_linear_predictions(
    fit: _Fit,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
) -> np.ndarray:
    """Extract draws on the approved model's linear-predictor scale."""
    return predict_model(
        fit,
        newdata=newdata,
        type="linear",
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
    ).draws


def predict_binary_probability(
    fit: BinaryFit,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> Prediction:
    """Return binary event-probability predictions."""
    validated = cast(BinaryFit, _validate_fit(fit, "binary"))
    return predict_model(
        validated,
        newdata=newdata,
        type="expected",
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
        probs=probs,
    )


def predict_duration(
    fit: DurationFit,
    newdata: pd.DataFrame | None = None,
    type: Literal["median", "expected", "predictive"] = "median",
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    seed: int = 1,
) -> Prediction:
    """Return duration predictions on the recorded response scale."""
    validated = cast(DurationFit, _validate_fit(fit, "duration"))
    if type not in {"median", "expected", "predictive"}:
        raise GP3BayesError(
            '`type` must be one of "median", "expected", or "predictive".'
        )
    return predict_model(
        validated,
        newdata=newdata,
        type=type,
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
        probs=probs,
        seed=seed,
    )
