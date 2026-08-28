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
    if pd.api.types.is_string_dtype(template.dtype) or pd.api.types.is_object_dtype(template.dtype):
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
            raise GP3BayesError("`variables` must identify columns in the prepared model data.")
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
            "Prediction data are missing fixed-effect variables: " + ", ".join(missing) + "."
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
            condition = pd.to_numeric(newdata[condition_col], errors="raise").to_numpy(dtype=float)
            participant_slope_cache: dict[str, np.ndarray] = {}
            for row, level in enumerate(requested):
                if level in lookup:
                    effect = participant_re[:, lookup[level], :]
                else:
                    if level not in participant_slope_cache:
                        sampled = rng.integers(0, participant_re.shape[1], size=ndraws)
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
        raise GP3BayesError(f"`ndraws` ({draw_n}) exceeds the {total} posterior draws available.")

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
                "  Include group effects: " + str(self.include_group_effects).upper(),
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
        prediction_data[outcome_col].copy() if outcome_col in prediction_data.columns else None
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
            sigma = _take_draws(_posterior_values(validated, "sigma"), eta.shape[0]).reshape(
                eta.shape[0], 1
            )
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
        raise GP3BayesError('`type` must be one of "median", "expected", or "predictive".')
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


def _prediction_inputs(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Normalize prediction-object or numeric-vector scoring inputs."""
    if isinstance(x, Prediction):
        if x.observed is None:
            raise GP3BayesError("The prediction object has no observed outcome.")
        return (
            x.summary["predicted_mean"].to_numpy(dtype=float, copy=True),
            pd.to_numeric(x.observed, errors="raise").to_numpy(dtype=float),
            np.asarray(x.draws, dtype=float),
        )

    try:
        predicted = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(
            "Supply a gp3bayes prediction, or numeric predictions plus `observed`."
        ) from exc
    if predicted.ndim != 1 or observed is None:
        raise GP3BayesError("Supply a gp3bayes prediction, or numeric predictions plus `observed`.")
    try:
        observed_values = np.asarray(observed, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(
            "Supply a gp3bayes prediction, or numeric predictions plus `observed`."
        ) from exc
    if observed_values.ndim != 1 or len(predicted) != len(observed_values):
        raise GP3BayesError("Supply a gp3bayes prediction, or numeric predictions plus `observed`.")
    return predicted, observed_values, None


def _finite_scalar(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be one finite number.")
    return float(value)


def _probability_vector(
    values: Sequence[float],
    name: str,
    *,
    open_interval: bool,
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(f"`{name}` must contain probabilities.") from exc
    if not result or any(not math.isfinite(value) for value in result):
        raise GP3BayesError(f"`{name}` must contain probabilities.")
    if open_interval:
        invalid = any(value <= 0 or value >= 1 for value in result)
        interval = "strictly inside (0,1)"
    else:
        invalid = any(value < 0 or value > 1 for value in result)
        interval = "in [0,1]"
    if invalid:
        raise GP3BayesError(f"`{name}` must contain probabilities {interval}.")
    return result


def binary_prediction_scores(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    threshold: float = 0.5,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Return descriptive binary predictive scores without an automatic decision."""
    p, y, _ = _prediction_inputs(x, observed)
    if (
        not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
        or not np.isfinite(y).all()
        or np.any(~np.isin(y, (0.0, 1.0)))
    ):
        raise GP3BayesError("Binary scores require outcomes in {0,1} and probabilities in [0,1].")
    threshold_value = _finite_scalar(threshold, "threshold")
    if threshold_value < 0 or threshold_value > 1:
        raise GP3BayesError("`threshold` must lie in [0, 1].")
    epsilon_value = _finite_scalar(epsilon, "epsilon")
    if epsilon_value <= 0 or epsilon_value >= 0.5:
        raise GP3BayesError("`epsilon` must be one finite number strictly inside (0, 0.5).")

    clipped = np.clip(p, epsilon_value, 1 - epsilon_value)
    predicted = (p >= threshold_value).astype(int)
    y_int = y.astype(int)
    true_positive = int(np.sum((predicted == 1) & (y_int == 1)))
    true_negative = int(np.sum((predicted == 0) & (y_int == 0)))
    false_positive = int(np.sum((predicted == 1) & (y_int == 0)))
    false_negative = int(np.sum((predicted == 0) & (y_int == 1)))
    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else math.nan
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else math.nan
    )

    n1 = int(np.sum(y_int == 1))
    n0 = int(np.sum(y_int == 0))
    if n1 > 0 and n0 > 0:
        ranks = pd.Series(p).rank(method="average").to_numpy(dtype=float)
        auc = float((np.sum(ranks[y_int == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        auc = math.nan

    balanced_parts = [value for value in (sensitivity, specificity) if math.isfinite(value)]
    balanced_accuracy = float(np.mean(balanced_parts)) if balanced_parts else math.nan
    return pd.DataFrame(
        [
            {
                "n": int(len(y_int)),
                "brier": float(np.mean((p - y) ** 2)),
                "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
                "auc": auc,
                "threshold": threshold_value,
                "accuracy": float(np.mean(predicted == y_int)),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "automatic_decision": False,
            }
        ]
    )


def binary_threshold_metrics(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    thresholds: Sequence[float] = tuple(np.arange(0.1, 0.9000001, 0.05)),
) -> pd.DataFrame:
    """Return binary classification summaries over declared thresholds."""
    values = _probability_vector(thresholds, "thresholds", open_interval=False)
    rows = [binary_prediction_scores(x, observed, threshold=threshold) for threshold in values]
    return pd.concat(rows, ignore_index=True)


def binary_calibration_table(
    x: Prediction,
    bins: int = 10,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Compare binary observed rates with posterior event probabilities by bin."""
    if (
        not isinstance(x, Prediction)
        or x.family != "binary"
        or x.type != "expected"
        or x.observed is None
    ):
        raise GP3BayesError("`x` must be a binary expected-response prediction with outcomes.")
    bins_value = _positive_integer(bins, "bins")
    if bins_value < 2:
        raise GP3BayesError("`bins` must be one integer >= 2.")
    probs_value = _probabilities(probs)
    pmean = x.summary["predicted_mean"].to_numpy(dtype=float)
    breaks = np.unique(
        np.quantile(
            pmean,
            np.linspace(0, 1, bins_value + 1),
            method="linear",
        )
    )
    if breaks.size < 3:
        bin_ids = np.ones(len(pmean), dtype=int)
    else:
        pmean_values = [float(value) for value in pmean]
        break_values = [float(value) for value in breaks]
        cut = pd.cut(
            pmean_values,
            bins=break_values,
            include_lowest=True,
            labels=False,
        )
        bin_ids = np.asarray(cut, dtype=int) + 1

    observed_values = pd.to_numeric(x.observed, errors="raise").to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    for bin_id in sorted(np.unique(bin_ids).tolist()):
        indices = np.flatnonzero(bin_ids == bin_id)
        draw_mean = np.mean(x.draws[:, indices], axis=1)
        quantiles = np.quantile(draw_mean, probs_value, method="linear")
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(len(indices)),
                "mean_predicted_probability": float(np.mean(pmean[indices])),
                "observed_rate": float(np.mean(observed_values[indices])),
                "posterior_lower": float(quantiles[0]),
                "posterior_median": float(quantiles[1]),
                "posterior_upper": float(quantiles[2]),
            }
        )
    return pd.DataFrame(rows)


def duration_prediction_scores(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
) -> pd.DataFrame:
    """Return descriptive duration prediction errors on response and log scales."""
    predicted, y, _ = _prediction_inputs(x, observed)
    if (
        not np.isfinite(predicted).all()
        or not np.isfinite(y).all()
        or np.any(predicted <= 0)
        or np.any(y <= 0)
    ):
        raise GP3BayesError("Duration scores require finite positive predictions and outcomes.")
    error = predicted - y
    log_error = np.log(predicted) - np.log(y)
    return pd.DataFrame(
        [
            {
                "n": int(len(y)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "median_absolute_error": float(np.median(np.abs(error))),
                "log_mae": float(np.mean(np.abs(log_error))),
                "log_rmse": float(np.sqrt(np.mean(log_error**2))),
                "mean_log_error": float(np.mean(log_error)),
                "automatic_decision": False,
            }
        ]
    )


def duration_quantile_calibration(
    x: Prediction,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> pd.DataFrame:
    """Compare nominal duration predictive quantiles with empirical coverage."""
    if (
        not isinstance(x, Prediction)
        or x.family != "duration"
        or x.type != "predictive"
        or x.observed is None
    ):
        raise GP3BayesError("`x` must be a duration posterior predictive object with outcomes.")
    values = _probability_vector(quantiles, "quantiles", open_interval=True)
    observed_values = pd.to_numeric(x.observed, errors="raise").to_numpy(dtype=float)
    rows = []
    for probability in values:
        predictive_quantile = np.quantile(
            x.draws,
            probability,
            axis=0,
            method="linear",
        )
        empirical = float(np.mean(observed_values <= predictive_quantile))
        rows.append(
            {
                "nominal": probability,
                "empirical": empirical,
                "calibration_gap": empirical - probability,
            }
        )
    return pd.DataFrame(rows)


def duration_pit_table(x: Prediction) -> pd.DataFrame:
    """Return empirical posterior-predictive PIT values for duration outcomes."""
    if (
        not isinstance(x, Prediction)
        or x.family != "duration"
        or x.type != "predictive"
        or x.observed is None
    ):
        raise GP3BayesError("`x` must be a duration posterior predictive object with outcomes.")
    observed_values = pd.to_numeric(x.observed, errors="raise").to_numpy(dtype=float)
    pit = np.mean(x.draws <= observed_values[None, :], axis=0)
    return pd.DataFrame(
        {
            "observation": np.arange(1, len(pit) + 1, dtype=int),
            "pit": pit,
        }
    )


def predictive_coverage_table(
    x: Prediction,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Return empirical coverage and width for posterior-predictive intervals."""
    if not isinstance(x, Prediction) or x.type != "predictive" or x.observed is None:
        raise GP3BayesError("`x` must be a posterior predictive object with observed outcomes.")
    coverage_levels = _probability_vector(levels, "levels", open_interval=True)
    observed_values = pd.to_numeric(x.observed, errors="raise").to_numpy(dtype=float)
    rows = []
    for level in coverage_levels:
        alpha = (1 - level) / 2
        lower = np.quantile(x.draws, alpha, axis=0, method="linear")
        upper = np.quantile(x.draws, 1 - alpha, axis=0, method="linear")
        rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage": float(
                    np.mean((observed_values >= lower) & (observed_values <= upper))
                ),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
        )
    return pd.DataFrame(rows)


def posterior_predictive_summary_table(
    x: Prediction,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Return observation-level summaries for posterior-predictive draws."""
    if not isinstance(x, Prediction) or x.type != "predictive":
        raise GP3BayesError("`x` must be a posterior predictive gp3bayes prediction.")
    return _prediction_summary(x.draws, _probabilities(probs), x.observed)


@dataclass(frozen=True, slots=True)
class _PredictionUncertainty:
    """Descriptive posterior Monte Carlo prediction-uncertainty decomposition."""

    table: pd.DataFrame
    expected: Prediction
    predictive: Prediction
    interpretation: str = (
        "Components are descriptive posterior Monte Carlo variances. "
        "They are not a causal variance decomposition."
    )
    causal_variance_decomposition: bool = False


@dataclass(frozen=True, slots=True)
class _GroupedPredictionCheck:
    """Grouped posterior-predictive summaries without automatic exclusion."""

    family: str
    group_column: str
    table: pd.DataFrame
    draws: np.ndarray
    automatic_exclusion: bool = False
    interpretation: str = (
        "Observed group summaries are compared with posterior predictive summaries. "
        "Large discrepancies request review; groups are not excluded automatically."
    )


def _prediction_row(value: object, name: str, n_rows: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) != math.floor(float(value))
        or float(value) < 1
        or float(value) > n_rows
    ):
        raise GP3BayesError("`row1` and `row2` must identify prediction rows.")
    return int(float(value))


def prediction_contrast(
    x: Prediction,
    row1: int,
    row2: int,
    measure: Literal["difference", "ratio", "odds_ratio"] = "difference",
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Summarize a posterior contrast between two 1-based prediction rows."""
    if not isinstance(x, Prediction):
        raise GP3BayesError("`x` must be a gp3bayes prediction.")
    if measure not in {"difference", "ratio", "odds_ratio"}:
        raise GP3BayesError('`measure` must be one of "difference", "ratio", or "odds_ratio".')
    row1_value = _prediction_row(row1, "row1", x.draws.shape[1])
    row2_value = _prediction_row(row2, "row2", x.draws.shape[1])
    a = np.asarray(x.draws[:, row1_value - 1], dtype=float)
    b = np.asarray(x.draws[:, row2_value - 1], dtype=float)

    if measure == "difference":
        values = b - a
        reference = 0.0
    elif measure == "ratio":
        if np.any(a <= 0):
            raise GP3BayesError("Ratio contrasts require positive denominator draws.")
        values = b / a
        reference = 1.0
    else:
        if x.family != "binary" or x.type != "expected":
            raise GP3BayesError("Odds-ratio contrasts require binary expected probabilities.")
        epsilon = math.sqrt(np.finfo(float).eps)
        a_clipped = np.clip(a, epsilon, 1 - epsilon)
        b_clipped = np.clip(b, epsilon, 1 - epsilon)
        values = (b_clipped / (1 - b_clipped)) / (a_clipped / (1 - a_clipped))
        reference = 1.0

    quantiles = np.quantile(values, _probabilities(probs), method="linear")
    return pd.DataFrame(
        [
            {
                "row1": row1_value,
                "row2": row2_value,
                "measure": measure,
                "mean": float(np.mean(values)),
                "lower": float(quantiles[0]),
                "median": float(quantiles[1]),
                "upper": float(quantiles[2]),
                "probability_gt_reference": float(np.mean(values > reference)),
                "automatic_decision": False,
            }
        ]
    )


def prediction_exceedance_probability(
    x: Prediction,
    threshold: float,
    direction: Literal["above", "below"] = "above",
) -> pd.DataFrame:
    """Return observation-level posterior exceedance probabilities."""
    if not isinstance(x, Prediction):
        raise GP3BayesError("`x` must be a gp3bayes prediction.")
    threshold_value = _finite_scalar(threshold, "threshold")
    if direction not in {"above", "below"}:
        raise GP3BayesError('`direction` must be either "above" or "below".')
    probability = (
        np.mean(x.draws > threshold_value, axis=0)
        if direction == "above"
        else np.mean(x.draws < threshold_value, axis=0)
    )
    return pd.DataFrame(
        {
            "observation": np.arange(1, len(probability) + 1, dtype=int),
            "threshold": threshold_value,
            "direction": direction,
            "probability": probability,
            "automatic_decision": False,
        }
    )


def prediction_uncertainty_decomposition(
    fit: _Fit,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int = 1000,
    seed: int = 1,
) -> _PredictionUncertainty:
    """Decompose predictive variability descriptively, not causally."""
    draw_n = _positive_integer(ndraws, "ndraws")
    seed_value = _nonnegative_integer(seed, "seed")
    expected = predict_model(
        fit,
        newdata=newdata,
        type="expected",
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=draw_n,
    )
    predictive = predict_model(
        fit,
        newdata=expected.newdata,
        type="predictive",
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=draw_n,
        seed=seed_value,
    )
    if draw_n > 1:
        expected_variance = np.var(expected.draws, axis=0, ddof=1)
        total_variance = np.var(predictive.draws, axis=0, ddof=1)
    else:
        expected_variance = np.full(expected.draws.shape[1], np.nan, dtype=float)
        total_variance = np.full(predictive.draws.shape[1], np.nan, dtype=float)
    residual = np.maximum(total_variance - expected_variance, 0.0)
    expected_fraction = np.divide(
        expected_variance,
        total_variance,
        out=np.full(total_variance.shape, np.nan, dtype=float),
        where=total_variance > 0,
    )
    table = pd.DataFrame(
        {
            "observation": np.arange(1, len(total_variance) + 1, dtype=int),
            "expected_response_variance": expected_variance,
            "total_predictive_variance": total_variance,
            "residual_component": residual,
            "expected_fraction": expected_fraction,
        }
    )
    return _PredictionUncertainty(
        table=table,
        expected=expected,
        predictive=predictive,
    )


def grouped_prediction_check(
    fit: _Fit,
    group: str,
    ndraws: int = 1000,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    seed: int = 1,
) -> _GroupedPredictionCheck:
    """Compare observed and posterior-predictive group means conservatively."""
    validated = _validate_fit(fit)
    if not isinstance(group, str) or not group:
        raise GP3BayesError("`group` must name one column in the prepared model data.")
    prepared = cast(Any, validated.specification.prepared)
    data = prepared.data
    if group not in data.columns:
        raise GP3BayesError("`group` must name one column in the prepared model data.")
    draw_n = _positive_integer(ndraws, "ndraws")
    probs_value = _probabilities(probs)
    seed_value = _nonnegative_integer(seed, "seed")
    prediction = predict_model(
        validated,
        type="predictive",
        include_group_effects=True,
        ndraws=draw_n,
        probs=probs_value,
        seed=seed_value,
    )
    outcome_col = cast(str, validated.specification.contract.mappings["outcome"])
    group_strings = data[group].astype(str)
    group_names = sorted(pd.unique(group_strings).tolist())
    group_draws = np.column_stack(
        [
            np.mean(
                prediction.draws[:, np.flatnonzero((group_strings == name).to_numpy())],
                axis=1,
            )
            for name in group_names
        ]
    )
    quantiles = np.quantile(group_draws, probs_value, axis=0, method="linear")
    observed_values = pd.to_numeric(data[outcome_col], errors="raise").to_numpy(dtype=float)
    rows = []
    for index, name in enumerate(group_names):
        members = np.flatnonzero((group_strings == name).to_numpy())
        rows.append(
            {
                "group": name,
                "n": int(len(members)),
                "observed": float(np.mean(observed_values[members])),
                "predicted_mean": float(np.mean(group_draws[:, index])),
                "lower": float(quantiles[0, index]),
                "predicted_median": float(quantiles[1, index]),
                "upper": float(quantiles[2, index]),
            }
        )
    return _GroupedPredictionCheck(
        family=validated.family,
        group_column=group,
        table=pd.DataFrame(rows),
        draws=group_draws,
    )


def predictive_residuals(
    fit: _Fit,
    type: Literal["raw", "pearson", "log", "relative"] | None = None,
    ndraws: int = 1000,
) -> pd.DataFrame:
    """Return descriptive residuals from posterior expected responses."""
    validated = _validate_fit(fit)
    if validated.family == "binary":
        residual_type = "raw" if type is None else type
        if residual_type not in {"raw", "pearson"}:
            raise GP3BayesError('Binary residual `type` must be either "raw" or "pearson".')
    else:
        residual_type = "log" if type is None else type
        if residual_type not in {"raw", "log", "relative"}:
            raise GP3BayesError(
                'Duration residual `type` must be one of "raw", "log", or "relative".'
            )
    draw_n = _positive_integer(ndraws, "ndraws")
    prediction = predict_model(
        validated,
        type="expected",
        include_group_effects=True,
        ndraws=draw_n,
    )
    if prediction.observed is None:
        raise GP3BayesError("Observed outcomes are unavailable.")
    observed = pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    expected = prediction.summary["predicted_mean"].to_numpy(dtype=float)

    if residual_type == "raw":
        residual = observed - expected
    elif residual_type == "pearson":
        denominator = np.sqrt(np.maximum(expected * (1 - expected), np.finfo(float).eps))
        residual = (observed - expected) / denominator
    elif residual_type == "log":
        residual = np.log(observed) - np.log(expected)
    else:
        residual = (observed - expected) / expected

    return pd.DataFrame(
        {
            "observation": np.arange(1, len(observed) + 1, dtype=int),
            "observed": observed,
            "expected": expected,
            "residual": residual,
            "type": residual_type,
        }
    )


@dataclass(frozen=True, slots=True)
class _PosteriorPredictiveStatistic:
    """Descriptive scalar posterior-predictive discrepancy."""

    family: str
    statistic: str
    threshold: float | None
    observed: float
    replicated: np.ndarray
    posterior_mean: float
    posterior_sd: float
    lower_tail_probability: float
    upper_tail_probability: float
    two_sided_tail_probability: float
    automatic_adequacy_verdict: bool = False
    interpretation: str = (
        "The tail probability is a descriptive posterior predictive discrepancy "
        "measure. It is not an automatic model-adequacy test."
    )


def _advanced_prediction(
    x: object,
    *,
    types: set[str] | None = None,
    family: str | None = None,
    observed: bool = False,
) -> Prediction:
    if not isinstance(x, Prediction):
        raise GP3BayesError("`x` must be a `gp3bayes_prediction`.")
    if types is not None and x.type not in types:
        allowed = ", ".join(sorted(types))
        raise GP3BayesError(f"`x.type` must be one of: {allowed}.")
    if family is not None and x.family != family:
        raise GP3BayesError(f"`x` must use the `{family}` family.")
    if observed and x.observed is None:
        raise GP3BayesError("Observed outcomes are required for this diagnostic.")
    return x


def _binary_probability_inputs(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(x, Prediction):
        prediction = _advanced_prediction(x, family="binary", observed=True)
        if prediction.type != "expected":
            raise GP3BayesError('Binary probability diagnostics require `type = "expected"`.')
        probabilities = prediction.summary["predicted_mean"].to_numpy(dtype=float)
        assert prediction.observed is not None
        outcomes = pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    else:
        if observed is None:
            raise GP3BayesError(
                "Supply a binary expected-response prediction or numeric probabilities "
                "plus numeric observed outcomes."
            )
        try:
            probabilities = np.asarray(x, dtype=float)
            outcomes = np.asarray(observed, dtype=float)
        except (TypeError, ValueError) as exc:
            raise GP3BayesError(
                "Supply a binary expected-response prediction or numeric probabilities "
                "plus numeric observed outcomes."
            ) from exc
        if probabilities.ndim != 1 or outcomes.ndim != 1 or len(probabilities) != len(outcomes):
            raise GP3BayesError(
                "Supply a binary expected-response prediction or numeric probabilities "
                "plus numeric observed outcomes."
            )
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(outcomes).all()
        or np.any((probabilities < 0) | (probabilities > 1))
        or np.any(~np.isin(outcomes, (0.0, 1.0)))
    ):
        raise GP3BayesError(
            "Binary diagnostics require finite probabilities from 0 to 1 and "
            "observed outcomes coded 0 or 1."
        )
    return probabilities, outcomes


def _advanced_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    try:
        probs = tuple(sorted(set(float(value) for value in values)))
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(
            "`probs` must contain finite probabilities strictly inside (0, 1)."
        ) from exc
    if not probs or any(not math.isfinite(value) or value <= 0 or value >= 1 for value in probs):
        raise GP3BayesError("`probs` must contain finite probabilities strictly inside (0, 1).")
    return probs


def prediction_draws_long(
    x: Prediction,
    max_draws: int | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    """Convert posterior prediction draws to observation-major long form."""
    prediction = _advanced_prediction(x)
    draws = np.asarray(prediction.draws, dtype=float)
    retained = _positive_integer_or_none(max_draws, "max_draws")
    if retained is not None and draws.shape[0] > retained:
        seed_value = _nonnegative_integer(seed, "seed")
        keep = np.sort(
            np.random.default_rng(seed_value).choice(draws.shape[0], size=retained, replace=False)
        )
        draws = draws[keep, :]
    return pd.DataFrame(
        {
            "draw": np.tile(np.arange(1, draws.shape[0] + 1, dtype=int), draws.shape[1]),
            "observation": np.repeat(np.arange(1, draws.shape[1] + 1, dtype=int), draws.shape[0]),
            "value": draws.reshape(-1, order="F"),
        }
    )


def _statistic_values(
    values: np.ndarray,
    statistic: str,
    threshold: float | None,
    *,
    axis: int | None,
) -> np.ndarray | float:
    if statistic == "mean":
        return np.mean(values, axis=axis)
    if statistic == "sd":
        return np.std(values, axis=axis, ddof=1)
    if statistic == "median":
        return np.median(values, axis=axis)
    if statistic == "q90":
        return np.quantile(values, 0.90, axis=axis, method="linear")
    if statistic == "q95":
        return np.quantile(values, 0.95, axis=axis, method="linear")
    if statistic == "max":
        return np.max(values, axis=axis)
    assert threshold is not None
    return cast(np.ndarray | float, np.mean(values > threshold, axis=axis))


def posterior_predictive_statistic(
    x: Prediction,
    statistic: Literal["mean", "sd", "median", "q90", "q95", "max", "tail_rate"] = "mean",
    threshold: float | None = None,
) -> _PosteriorPredictiveStatistic:
    """Compare one observed discrepancy with its posterior-predictive distribution."""
    prediction = _advanced_prediction(x, types={"predictive"}, observed=True)
    allowed = {"mean", "sd", "median", "q90", "q95", "max", "tail_rate"}
    if statistic not in allowed:
        raise GP3BayesError("Unsupported posterior-predictive statistic.")
    threshold_value: float | None = None
    if statistic == "tail_rate":
        if threshold is None:
            raise GP3BayesError(
                '`threshold` must be one finite number when `statistic = "tail_rate"`.'
            )
        threshold_value = _finite_scalar(threshold, "threshold")
    assert prediction.observed is not None
    observed_values = pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    observed_value = float(
        _statistic_values(observed_values, statistic, threshold_value, axis=None)
    )
    replicated = np.asarray(
        _statistic_values(
            np.asarray(prediction.draws, dtype=float),
            statistic,
            threshold_value,
            axis=1,
        ),
        dtype=float,
    )
    if not math.isfinite(observed_value) or not np.isfinite(replicated).all():
        raise GP3BayesError("The selected predictive statistic produced non-finite values.")
    upper = float(np.mean(replicated >= observed_value))
    lower = float(np.mean(replicated <= observed_value))
    return _PosteriorPredictiveStatistic(
        family=prediction.family,
        statistic=statistic,
        threshold=threshold_value,
        observed=observed_value,
        replicated=replicated,
        posterior_mean=float(np.mean(replicated)),
        posterior_sd=float(np.std(replicated, ddof=1)),
        lower_tail_probability=lower,
        upper_tail_probability=upper,
        two_sided_tail_probability=min(1.0, 2.0 * min(upper, lower)),
    )


def ppc_statistic_table(x: _PosteriorPredictiveStatistic) -> pd.DataFrame:
    """Return the one-row descriptive posterior-predictive statistic table."""
    if not isinstance(x, _PosteriorPredictiveStatistic):
        raise GP3BayesError("`x` must be a `gp3bayes_ppc_statistic`.")
    return pd.DataFrame(
        [
            {
                "statistic": x.statistic,
                "threshold": math.nan if x.threshold is None else x.threshold,
                "observed": x.observed,
                "posterior_mean": x.posterior_mean,
                "posterior_sd": x.posterior_sd,
                "lower_tail_probability": x.lower_tail_probability,
                "upper_tail_probability": x.upper_tail_probability,
                "two_sided_tail_probability": x.two_sided_tail_probability,
                "automatic_adequacy_verdict": False,
            }
        ]
    )


def binary_confusion_table(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return the fixed four-cell binary confusion table."""
    probabilities, outcomes = _binary_probability_inputs(x, observed)
    threshold_value = _finite_scalar(threshold, "threshold")
    if threshold_value < 0 or threshold_value > 1:
        raise GP3BayesError("`threshold` must be one finite number from 0 to 1.")
    predicted = (probabilities >= threshold_value).astype(int)
    y = outcomes.astype(int)
    counts = [
        int(np.sum((y == 0) & (predicted == 0))),
        int(np.sum((y == 0) & (predicted == 1))),
        int(np.sum((y == 1) & (predicted == 0))),
        int(np.sum((y == 1) & (predicted == 1))),
    ]
    return pd.DataFrame(
        {
            "observed": [0, 0, 1, 1],
            "predicted": [0, 1, 0, 1],
            "count": counts,
            "threshold": threshold_value,
        }
    )


def _binary_curve_thresholds(
    probabilities: np.ndarray,
    thresholds: Sequence[float] | np.ndarray[Any, Any] | None,
) -> np.ndarray:
    if thresholds is None:
        values = np.concatenate(([math.inf], probabilities, [-math.inf]))
    else:
        try:
            values = np.asarray(thresholds, dtype=float)
        except (TypeError, ValueError) as exc:
            raise GP3BayesError(
                "`thresholds` must be NULL or numeric without missing values."
            ) from exc
        if values.ndim != 1 or np.isnan(values).any():
            raise GP3BayesError("`thresholds` must be NULL or numeric without missing values.")
    return np.asarray(sorted(set(values.tolist()), reverse=True), dtype=float)


def binary_roc_curve(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    thresholds: Sequence[float] | np.ndarray[Any, Any] | None = None,
) -> pd.DataFrame:
    """Return deterministic empirical ROC coordinates over declared thresholds."""
    probabilities, outcomes = _binary_probability_inputs(x, observed)
    threshold_values = _binary_curve_thresholds(probabilities, thresholds)
    positives = int(np.sum(outcomes == 1))
    negatives = int(np.sum(outcomes == 0))
    rows = []
    for threshold in threshold_values:
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & (outcomes == 1)))
        false_positive = int(np.sum(predicted & (outcomes == 0)))
        rows.append(
            {
                "threshold": float(threshold),
                "false_positive_rate": (false_positive / negatives if negatives else math.nan),
                "true_positive_rate": (true_positive / positives if positives else math.nan),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["false_positive_rate", "true_positive_rate"],
        na_position="last",
        kind="stable",
        ignore_index=True,
    )


def binary_precision_recall_curve(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    thresholds: Sequence[float] | np.ndarray[Any, Any] | None = None,
) -> pd.DataFrame:
    """Return deterministic empirical precision-recall coordinates."""
    probabilities, outcomes = _binary_probability_inputs(x, observed)
    threshold_values = _binary_curve_thresholds(probabilities, thresholds)
    positives = int(np.sum(outcomes == 1))
    rows = []
    for threshold in threshold_values:
        predicted = probabilities >= threshold
        true_positive = int(np.sum(predicted & (outcomes == 1)))
        false_positive = int(np.sum(predicted & (outcomes == 0)))
        rows.append(
            {
                "threshold": float(threshold),
                "recall": true_positive / positives if positives else math.nan,
                "precision": (
                    true_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else 1.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["recall", "precision"],
        na_position="last",
        kind="stable",
        ignore_index=True,
    )


def binary_calibration_error(
    x: Prediction | Sequence[float] | np.ndarray[Any, Any] | pd.Series,
    observed: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Return equal-frequency expected and maximum absolute calibration error."""
    probabilities, outcomes = _binary_probability_inputs(x, observed)
    bins_value = _positive_integer(bins, "bins")
    if bins_value < 2:
        raise GP3BayesError("`bins` must be one integer greater than or equal to 2.")
    breaks = np.unique(
        np.quantile(
            probabilities,
            np.linspace(0, 1, bins_value + 1),
            method="median_unbiased",
        )
    )
    if len(breaks) < 3:
        bin_ids = np.ones(len(probabilities), dtype=int)
    else:
        bin_ids = np.digitize(probabilities, breaks[1:-1], right=True) + 1
    groups = [np.flatnonzero(bin_ids == value) for value in sorted(set(bin_ids.tolist()))]
    weights = np.asarray([len(index) / len(probabilities) for index in groups])
    gaps = np.asarray(
        [
            abs(float(np.mean(probabilities[index])) - float(np.mean(outcomes[index])))
            for index in groups
        ]
    )
    return pd.DataFrame(
        [
            {
                "n": int(len(probabilities)),
                "bins_requested": bins_value,
                "bins_used": int(len(groups)),
                "expected_calibration_error": float(np.sum(weights * gaps)),
                "maximum_calibration_error": float(np.max(gaps)),
                "automatic_adequacy_verdict": False,
            }
        ]
    )


def binary_group_calibration(x: Prediction, group: str) -> pd.DataFrame:
    """Summarize binary expected-probability calibration by newdata group."""
    prediction = _advanced_prediction(x, types={"expected"}, family="binary", observed=True)
    if not isinstance(group, str) or group not in prediction.newdata.columns:
        raise GP3BayesError("`group` must name one column in `x.newdata`.")
    assert prediction.observed is not None
    labels = prediction.newdata[group].astype(str).to_numpy()
    probabilities = prediction.summary["predicted_mean"].to_numpy(dtype=float)
    outcomes = pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    rows = []
    for label in sorted(set(labels.tolist())):
        index = np.flatnonzero(labels == label)
        predicted_probability = float(np.mean(probabilities[index]))
        observed_rate = float(np.mean(outcomes[index]))
        rows.append(
            {
                "group": label,
                "n": int(len(index)),
                "predicted_probability": predicted_probability,
                "observed_rate": observed_rate,
                "calibration_gap": observed_rate - predicted_probability,
            }
        )
    return pd.DataFrame(rows)


def duration_qq_table(
    x: Prediction,
    probs: Sequence[float] = tuple(np.arange(0.05, 0.951, 0.05)),
) -> pd.DataFrame:
    """Compare observed duration quantiles with predictive-draw quantiles."""
    prediction = _advanced_prediction(x, types={"predictive"}, family="duration", observed=True)
    probabilities = _advanced_probabilities(probs)
    assert prediction.observed is not None
    observed = pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    observed_quantiles = np.quantile(observed, probabilities, method="linear")
    predictive_quantiles = np.quantile(prediction.draws, probabilities, axis=1, method="linear")
    return pd.DataFrame(
        {
            "probability": probabilities,
            "observed_quantile": observed_quantiles,
            "predictive_mean_quantile": np.mean(predictive_quantiles, axis=1),
            "predictive_lower_quantile": np.quantile(
                predictive_quantiles, 0.025, axis=1, method="linear"
            ),
            "predictive_upper_quantile": np.quantile(
                predictive_quantiles, 0.975, axis=1, method="linear"
            ),
        }
    )


def duration_tail_check(x: Prediction, threshold: float) -> pd.DataFrame:
    """Compare observed and posterior-predictive duration tail rates."""
    prediction = _advanced_prediction(x, types={"predictive"}, family="duration", observed=True)
    threshold_value = _finite_scalar(threshold, "threshold")
    if threshold_value <= 0:
        raise GP3BayesError("`threshold` must be one finite positive duration.")
    assert prediction.observed is not None
    observed = pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    replicated_rates = np.mean(prediction.draws > threshold_value, axis=1)
    observed_rate = float(np.mean(observed > threshold_value))
    return pd.DataFrame(
        [
            {
                "threshold": threshold_value,
                "observed_tail_rate": observed_rate,
                "predictive_mean_tail_rate": float(np.mean(replicated_rates)),
                "predictive_lower_tail_rate": float(
                    np.quantile(replicated_rates, 0.025, method="linear")
                ),
                "predictive_upper_tail_rate": float(
                    np.quantile(replicated_rates, 0.975, method="linear")
                ),
                "posterior_probability_rate_ge_observed": float(
                    np.mean(replicated_rates >= observed_rate)
                ),
                "automatic_adequacy_verdict": False,
            }
        ]
    )


def group_prediction_summary(
    x: Prediction,
    by: str | Sequence[str],
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Aggregate prediction draws across one or more columns in newdata."""
    prediction = _advanced_prediction(x)
    columns = [by] if isinstance(by, str) else list(by)
    if (
        not columns
        or any(not isinstance(column, str) for column in columns)
        or any(column not in prediction.newdata.columns for column in columns)
    ):
        raise GP3BayesError("`by` must name one or more columns in `x.newdata`.")
    probabilities = _probabilities(probs)
    key_data = prediction.newdata[columns].copy()
    grouped = key_data.groupby(columns, sort=True, dropna=False).indices
    observed_values = (
        None
        if prediction.observed is None
        else pd.to_numeric(prediction.observed, errors="raise").to_numpy(dtype=float)
    )
    rows: list[dict[str, Any]] = []
    for _, raw_index in grouped.items():
        index = np.asarray(raw_index, dtype=int)
        group_draws = np.mean(prediction.draws[:, index], axis=1)
        quantiles = np.quantile(group_draws, probabilities, method="linear")
        identity = {column: key_data.iloc[index[0]][column] for column in columns}
        rows.append(
            {
                **identity,
                "n": int(len(index)),
                "predicted_mean": float(np.mean(group_draws)),
                "lower": float(quantiles[0]),
                "predicted_median": float(quantiles[1]),
                "upper": float(quantiles[2]),
                "observed": (
                    math.nan if observed_values is None else float(np.mean(observed_values[index]))
                ),
            }
        )
    return pd.DataFrame(rows)


def _prediction_rows(
    rows: Sequence[int] | np.ndarray[Any, Any] | None,
    n_rows: int,
) -> list[int]:
    if rows is None:
        return list(range(1, n_rows + 1))
    try:
        values = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError("`rows` must contain valid prediction-row indices.") from exc
    if (
        values.ndim != 1
        or np.isnan(values).any()
        or not np.isfinite(values).all()
        or np.any(values != np.floor(values))
        or np.any((values < 1) | (values > n_rows))
    ):
        raise GP3BayesError("`rows` must contain valid prediction-row indices.")
    return list(dict.fromkeys(int(value) for value in values.tolist()))


def prediction_pairwise_contrasts(
    x: Prediction,
    rows: Sequence[int] | np.ndarray[Any, Any] | None = None,
    measure: Literal["difference", "ratio"] = "difference",
    max_rows: int = 20,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    """Return every unique pairwise contrast among explicitly bounded rows."""
    prediction = _advanced_prediction(x)
    if measure not in {"difference", "ratio"}:
        raise GP3BayesError('`measure` must be either "difference" or "ratio".')
    selected = _prediction_rows(rows, prediction.draws.shape[1])
    limit = _positive_integer(max_rows, "max_rows")
    if limit < 2:
        raise GP3BayesError("`max_rows` must be one integer greater than or equal to 2.")
    if len(selected) > limit:
        raise GP3BayesError(
            f"Requested {len(selected)} prediction rows; the explicit maximum is {limit}."
        )
    if len(selected) < 2:
        raise GP3BayesError("At least two prediction rows are required for pairwise contrasts.")
    results = [
        prediction_contrast(
            prediction,
            row1=selected[first],
            row2=selected[second],
            measure=measure,
            probs=probs,
        )
        for first in range(len(selected) - 1)
        for second in range(first + 1, len(selected))
    ]
    return pd.concat(results, ignore_index=True)


def prediction_interval_width(x: Prediction) -> pd.DataFrame:
    """Return posterior interval width by prediction observation."""
    prediction = _advanced_prediction(x)
    summary = prediction.summary
    return pd.DataFrame(
        {
            "observation": summary["observation"].to_numpy(copy=True),
            "lower": summary["lower"].to_numpy(dtype=float, copy=True),
            "upper": summary["upper"].to_numpy(dtype=float, copy=True),
            "interval_width": (
                summary["upper"].to_numpy(dtype=float) - summary["lower"].to_numpy(dtype=float)
            ),
            "predicted_mean": summary["predicted_mean"].to_numpy(dtype=float, copy=True),
        }
    )


def prediction_rank_probabilities(
    x: Prediction,
    rows: Sequence[int] | np.ndarray[Any, Any] | None = None,
    direction: Literal["higher", "lower"] = "higher",
    max_rows: int = 20,
) -> pd.DataFrame:
    """Summarize relative ranks without selecting a prediction row automatically."""
    prediction = _advanced_prediction(x)
    if direction not in {"higher", "lower"}:
        raise GP3BayesError('`direction` must be either "higher" or "lower".')
    selected_rows = _prediction_rows(rows, prediction.draws.shape[1])
    limit = _positive_integer(max_rows, "max_rows")
    if len(selected_rows) > limit:
        raise GP3BayesError("Too many rows requested for ranking; increase `max_rows` explicitly.")
    selected = prediction.draws[:, np.asarray(selected_rows, dtype=int) - 1]
    ranks = np.empty_like(selected, dtype=float)
    for draw_index, values in enumerate(selected):
        ranked = -values if direction == "higher" else values
        ranks[draw_index, :] = pd.Series(ranked).rank(method="average").to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "observation": selected_rows,
            "probability_rank_1": np.mean(ranks == 1, axis=0),
            "mean_rank": np.mean(ranks, axis=0),
            "median_rank": np.median(ranks, axis=0),
            "automatic_selection": False,
        }
    )


# ---------------------------------------------------------------------------
# Frozen gp3bayes 0.5.0 prediction profiles, surfaces, atlases, and graphics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _PredictionProfile:
    variable: str
    table: pd.DataFrame
    prediction: Prediction
    automatic_effect_interpretation: bool = False
    interpretation: str = (
        "The profile is a fitted predictive description across requested values. "
        "It is not a causal response curve."
    )


@dataclass(frozen=True, slots=True)
class _PredictionSurface:
    x: str
    y: str
    table: pd.DataFrame
    prediction: Prediction
    automatic_interaction_decision: bool = False
    interpretation: str = (
        "The surface visualizes fitted predictive structure across a declared grid. "
        "It does not establish a causal interaction."
    )


@dataclass(frozen=True, slots=True)
class _PredictionContrastProfile:
    variable: str
    contrast_variable: str
    contrast_levels: tuple[Any, Any]
    measure: str
    table: pd.DataFrame
    draws: np.ndarray
    prediction: Prediction
    automatic_interaction_decision: bool = False
    interpretation: str = (
        "The contrast profile is a fitted posterior comparison across requested "
        "values and does not establish a causal interaction."
    )


@dataclass(frozen=True, slots=True)
class _PredictiveDistributionAtlas:
    family: str
    prediction: Prediction
    observed: np.ndarray
    observed_statistics: Mapping[str, float]
    draw_statistics: pd.DataFrame
    include_group_effects: bool
    automatic_adequacy_decision: bool = False
    interpretation: str = (
        "Observed distribution summaries are compared with posterior predictive "
        "replicates without an automatic adequacy verdict."
    )


@dataclass(frozen=True, slots=True)
class _PredictionScoreUncertainty:
    family: str
    scope: str
    draws: pd.DataFrame
    summary: pd.DataFrame
    prediction: Prediction
    automatic_model_ranking: bool = False
    interpretation: str = (
        "Score distributions propagate posterior uncertainty on the supplied "
        "evaluation data and are not automatically out-of-sample estimates."
    )


@dataclass(frozen=True, slots=True)
class _BinaryCalibrationUncertainty:
    table: pd.DataFrame
    bins_requested: int
    prediction: Prediction
    scope: str
    automatic_calibration_decision: bool = False
    interpretation: str = (
        "Observed bin rates are compared with posterior uncertainty in predicted "
        "probabilities without automatic calibration certification."
    )


def _profile_numeric(fit: _Fit, variable: str) -> pd.Series:
    validated = _validate_fit(fit)
    data = cast(Any, validated.specification.prepared).data
    if not isinstance(variable, str) or variable not in data.columns:
        raise GP3BayesError("`variable` must name one prepared-data column.")
    if not pd.api.types.is_numeric_dtype(data[variable]):
        raise GP3BayesError(f"`{variable}` must be numeric.")
    return data[variable]


def _profile_values(
    template: pd.Series,
    values: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None,
    n: int,
    variable: str,
) -> np.ndarray:
    if values is None:
        count = _positive_integer(n, "n")
        if count < 2:
            raise GP3BayesError("`n` must be one integer greater than or equal to 2.")
        numeric = pd.to_numeric(template, errors="coerce").to_numpy(dtype=float)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0 or float(np.min(finite)) == float(np.max(finite)):
            raise GP3BayesError(f"`{variable}` has no usable numeric range.")
        result = np.linspace(float(np.min(finite)), float(np.max(finite)), count)
    else:
        try:
            result = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise GP3BayesError("Profile values must be finite numeric values.") from exc
        if result.size < 2 or not np.isfinite(result).all():
            raise GP3BayesError("Profile values must be finite numeric values.")
    return np.unique(np.sort(result.astype(float)))


def _named_at(at: Mapping[str, object] | None) -> dict[str, object]:
    if at is None:
        return {}
    if not isinstance(at, Mapping) or any(not isinstance(k, str) or not k for k in at):
        raise GP3BayesError("`at` must be a named mapping.")
    return dict(at)


def create_prediction_profile(
    fit: _Fit,
    variable: str,
    values: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    n: int = 50,
    at: Mapping[str, object] | None = None,
    type: _PredictionType = "expected",
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    seed: int = 1,
) -> _PredictionProfile:
    """Create the frozen R 0.5.0 numeric posterior-prediction profile."""
    template = _profile_numeric(fit, variable)
    profile_values = _profile_values(template, values, n, variable)
    at_map = _named_at(at)
    at_map[variable] = profile_values
    grid = (
        create_prediction_grid(
            fit, variables=variable, at=at_map, max_rows=max(5000, len(profile_values))
        )
        .sort_values(variable, kind="stable")
        .reset_index(drop=True)
    )
    pred = predict_model(
        fit,
        newdata=grid,
        type=type,
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
        probs=probs,
        seed=seed,
    )
    table = pred.summary.copy()
    table["profile_x"] = grid[variable].to_numpy(dtype=float)
    return _PredictionProfile(variable=variable, table=table, prediction=pred)


def prediction_profile_table(x: _PredictionProfile) -> pd.DataFrame:
    if not isinstance(x, _PredictionProfile):
        raise GP3BayesError("`x` must be a gp3bayes prediction profile.")
    return x.table.copy()


def prediction_gradient_table(
    x: _PredictionProfile,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    if not isinstance(x, _PredictionProfile):
        raise GP3BayesError("`x` must be a gp3bayes prediction profile.")
    qprobs = _probabilities(probs)
    values = x.table["profile_x"].to_numpy(dtype=float)
    draws = np.asarray(x.prediction.draws, dtype=float)
    if values.size != draws.shape[1] or np.any(np.diff(values) <= 0):
        raise GP3BayesError("Prediction-profile values must be strictly increasing.")
    slopes = np.diff(draws, axis=1) / np.diff(values)[None, :]
    quant = np.quantile(slopes, qprobs, axis=0, method="linear")
    return pd.DataFrame(
        {
            "variable": x.variable,
            "lower_x": values[:-1],
            "upper_x": values[1:],
            "gradient_midpoint": (values[:-1] + values[1:]) / 2,
            "gradient_mean": np.mean(slopes, axis=0),
            "gradient_lower": quant[0],
            "gradient_median": quant[1],
            "gradient_upper": quant[2],
            "automatic_monotonicity_decision": False,
        }
    )


def create_prediction_surface(
    fit: _Fit,
    x: str,
    y: str,
    x_values: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    y_values: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    n: int = 30,
    at: Mapping[str, object] | None = None,
    type: _PredictionType = "expected",
    include_group_effects: bool = False,
    allow_new_levels: bool = False,
    ndraws: int | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    seed: int = 1,
    max_rows: int = 2500,
) -> _PredictionSurface:
    if x == y:
        raise GP3BayesError("`x` and `y` must differ.")
    xv = _profile_values(_profile_numeric(fit, x), x_values, n, x)
    yv = _profile_values(_profile_numeric(fit, y), y_values, n, y)
    limit = _positive_integer(max_rows, "max_rows")
    if limit < 4:
        raise GP3BayesError("`max_rows` must be one integer greater than or equal to 4.")
    if xv.size * yv.size > limit:
        raise GP3BayesError("Prediction surface exceeds `max_rows`.")
    at_map = _named_at(at)
    at_map[x] = xv
    at_map[y] = yv
    grid = create_prediction_grid(fit, variables=(x, y), at=at_map, max_rows=limit)
    pred = predict_model(
        fit,
        newdata=grid,
        type=type,
        include_group_effects=include_group_effects,
        allow_new_levels=allow_new_levels,
        ndraws=ndraws,
        probs=probs,
        seed=seed,
    )
    table = pred.summary.copy()
    table["surface_x"] = pd.to_numeric(grid[x], errors="raise").to_numpy(dtype=float)
    table["surface_y"] = pd.to_numeric(grid[y], errors="raise").to_numpy(dtype=float)
    table["interval_width"] = table["upper"] - table["lower"]
    return _PredictionSurface(x=x, y=y, table=table, prediction=pred)


def prediction_surface_table(x: _PredictionSurface) -> pd.DataFrame:
    if not isinstance(x, _PredictionSurface):
        raise GP3BayesError("`x` must be a gp3bayes prediction surface.")
    return x.table.copy()


def create_prediction_contrast_profile(
    fit: _Fit,
    variable: str,
    contrast_variable: str,
    contrast_levels: Sequence[object] | None = None,
    values: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
    n: int = 40,
    at: Mapping[str, object] | None = None,
    measure: Literal["difference", "ratio", "odds_ratio"] = "difference",
    include_group_effects: bool = False,
    ndraws: int | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> _PredictionContrastProfile:
    template = _profile_numeric(fit, variable)
    validated = _validate_fit(fit)
    data = cast(Any, validated.specification.prepared).data
    if not isinstance(contrast_variable, str) or contrast_variable not in data.columns:
        raise GP3BayesError("`contrast_variable` must name one prepared-data column.")
    pvalues = _profile_values(template, values, n, variable)
    if measure not in {"difference", "ratio", "odds_ratio"}:
        raise GP3BayesError("`measure` must be difference, ratio, or odds_ratio.")
    qprobs = _probabilities(probs)
    series = data[contrast_variable]
    if contrast_levels is None:
        observed_levels = (
            list(series.cat.categories)
            if isinstance(series.dtype, pd.CategoricalDtype)
            else list(pd.unique(series.astype(str)))
        )
        if len(observed_levels) != 2:
            raise GP3BayesError(
                "`contrast_levels` is required unless exactly two levels are observed."
            )
        levels = tuple(observed_levels)
    else:
        levels_list = list(contrast_levels)
        if len(levels_list) != 2 or any(pd.isna(v) for v in levels_list):  # type: ignore[call-overload]
            raise GP3BayesError("`contrast_levels` must contain exactly two values.")
        levels = (levels_list[0], levels_list[1])
    at_map = _named_at(at)
    at_map[variable] = pvalues
    at_map[contrast_variable] = list(levels)
    grid = create_prediction_grid(
        fit,
        variables=(variable, contrast_variable),
        at=at_map,
        max_rows=max(5000, 2 * len(pvalues)),
    )
    pred = predict_model(
        fit,
        newdata=grid,
        type="expected",
        include_group_effects=include_group_effects,
        allow_new_levels=False,
        ndraws=ndraws,
        probs=qprobs,
    )
    contrast_text = grid[contrast_variable].astype(str).to_numpy()
    x_grid = pd.to_numeric(grid[variable], errors="raise").to_numpy(dtype=float)
    contrast_draws = np.empty((pred.draws.shape[0], len(pvalues)), dtype=float)
    for idx, value in enumerate(pvalues):
        a_rows = np.flatnonzero((x_grid == value) & (contrast_text == str(levels[0])))
        b_rows = np.flatnonzero((x_grid == value) & (contrast_text == str(levels[1])))
        if a_rows.size != 1 or b_rows.size != 1:
            raise GP3BayesError("Contrast grid did not produce one row per level/value.")
        a = pred.draws[:, a_rows[0]]
        b = pred.draws[:, b_rows[0]]
        if measure == "difference":
            result = b - a
        elif measure == "ratio":
            if np.any(a <= 0):
                raise GP3BayesError("Ratio contrasts require positive denominators.")
            result = b / a
        else:
            if validated.family != "binary":
                raise GP3BayesError("Odds-ratio profiles require a binary fit.")
            eps = math.sqrt(np.finfo(float).eps)
            ac = np.clip(a, eps, 1 - eps)
            bc = np.clip(b, eps, 1 - eps)
            result = (bc / (1 - bc)) / (ac / (1 - ac))
        contrast_draws[:, idx] = result
    quant = np.quantile(contrast_draws, qprobs, axis=0, method="linear")
    reference = 0.0 if measure == "difference" else 1.0
    table = pd.DataFrame(
        {
            "profile_x": pvalues,
            "contrast_level_1": str(levels[0]),
            "contrast_level_2": str(levels[1]),
            "measure": measure,
            "contrast_mean": np.mean(contrast_draws, axis=0),
            "contrast_lower": quant[0],
            "contrast_median": quant[1],
            "contrast_upper": quant[2],
            "probability_gt_reference": np.mean(contrast_draws > reference, axis=0),
            "automatic_interaction_decision": False,
        }
    )
    return _PredictionContrastProfile(
        variable=variable,
        contrast_variable=contrast_variable,
        contrast_levels=levels,
        measure=measure,
        table=table,
        draws=contrast_draws,
        prediction=pred,
    )


def prediction_contrast_profile_table(x: _PredictionContrastProfile) -> pd.DataFrame:
    if not isinstance(x, _PredictionContrastProfile):
        raise GP3BayesError("`x` must be a gp3bayes prediction contrast profile.")
    return x.table.copy()


def _atlas_stat(values: np.ndarray) -> dict[str, float]:
    z = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(z)),
        "sd": float(np.std(z, ddof=1)) if z.size > 1 else float("nan"),
        "median": float(np.median(z)),
        "q10": float(np.quantile(z, 0.10, method="linear")),
        "q90": float(np.quantile(z, 0.90, method="linear")),
    }


def create_predictive_distribution_atlas(
    fit: _Fit,
    ndraws: int = 500,
    include_group_effects: bool = True,
    seed: int = 1,
) -> _PredictiveDistributionAtlas:
    validated = _validate_fit(fit)
    pred = predict_model(
        validated,
        type="predictive",
        include_group_effects=include_group_effects,
        allow_new_levels=False,
        ndraws=ndraws,
        seed=seed,
    )
    prepared = cast(Any, validated.specification.prepared)
    outcome = cast(str, validated.specification.contract.mappings["outcome"])
    observed = pd.to_numeric(prepared.data[outcome], errors="raise").to_numpy(dtype=float)
    rows = [_atlas_stat(row) for row in pred.draws]
    stats = pd.DataFrame(rows)
    stats.insert(0, "draw", np.arange(1, len(stats) + 1, dtype=int))
    return _PredictiveDistributionAtlas(
        family=validated.family,
        prediction=pred,
        observed=observed,
        observed_statistics=_atlas_stat(observed),
        draw_statistics=stats,
        include_group_effects=bool(include_group_effects),
    )


def predictive_distribution_atlas_table(x: _PredictiveDistributionAtlas) -> pd.DataFrame:
    if not isinstance(x, _PredictiveDistributionAtlas):
        raise GP3BayesError("`x` must be a gp3bayes predictive distribution atlas.")
    return x.draw_statistics.copy()


def _atlas_get(
    x: _Fit | _PredictiveDistributionAtlas,
    ndraws: int,
    include_group_effects: bool,
    seed: int,
) -> _PredictiveDistributionAtlas:
    if isinstance(x, _PredictiveDistributionAtlas):
        return x
    if isinstance(x, (BinaryFit, DurationFit)):
        return create_predictive_distribution_atlas(
            x,
            ndraws=ndraws,
            include_group_effects=include_group_effects,
            seed=seed,
        )
    raise GP3BayesError("`x` must be a fitted gp3bayes model or predictive atlas.")


def predictive_quantile_envelope(
    x: _Fit | _PredictiveDistributionAtlas,
    probabilities: Sequence[float] = tuple(np.arange(0.05, 1.0, 0.05)),
    probs: Sequence[float] = (0.025, 0.5, 0.975),
    ndraws: int = 500,
    include_group_effects: bool = True,
    seed: int = 1,
) -> pd.DataFrame:
    try:
        probabilities_value = sorted(set(float(v) for v in probabilities))
    except (TypeError, ValueError) as exc:
        raise GP3BayesError("`probabilities` must lie strictly between 0 and 1.") from exc
    if not probabilities_value or any(
        not math.isfinite(v) or v <= 0 or v >= 1 for v in probabilities_value
    ):
        raise GP3BayesError("`probabilities` must lie strictly between 0 and 1.")
    qprobs = _probabilities(probs)
    atlas = _atlas_get(x, ndraws, include_group_effects, seed)
    rows: list[dict[str, float]] = []
    for probability in probabilities_value:
        replicated = np.quantile(atlas.prediction.draws, probability, axis=1, method="linear")
        q = np.quantile(replicated, qprobs, method="linear")
        rows.append(
            {
                "probability": probability,
                "observed_quantile": float(
                    np.quantile(atlas.observed, probability, method="linear")
                ),
                "predictive_mean": float(np.mean(replicated)),
                "predictive_lower": float(q[0]),
                "predictive_median": float(q[1]),
                "predictive_upper": float(q[2]),
            }
        )
    return pd.DataFrame(rows)


def prediction_score_uncertainty(
    fit: _Fit,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = False,
    ndraws: int = 1000,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> _PredictionScoreUncertainty:
    validated = _validate_fit(fit)
    qprobs = _probabilities(probs)
    prepared = cast(Any, validated.specification.prepared)
    data = prepared.data.copy() if newdata is None else newdata.copy()
    outcome = cast(str, validated.specification.contract.mappings["outcome"])
    if not isinstance(data, pd.DataFrame) or outcome not in data.columns:
        raise GP3BayesError("Prediction-score uncertainty requires observed outcomes.")
    observed = pd.to_numeric(data[outcome], errors="raise").to_numpy(dtype=float)
    pred = predict_model(
        validated,
        newdata=data,
        type="expected",
        include_group_effects=include_group_effects,
        allow_new_levels=False,
        ndraws=ndraws,
    )
    obs_matrix = np.broadcast_to(observed[None, :], pred.draws.shape)
    if validated.family == "binary":
        eps = math.sqrt(np.finfo(float).eps)
        probabilities = np.clip(pred.draws, eps, 1 - eps)
        metric_draws = {
            "brier": np.mean((probabilities - obs_matrix) ** 2, axis=1),
            "log_loss": -np.mean(
                obs_matrix * np.log(probabilities) + (1 - obs_matrix) * np.log(1 - probabilities),
                axis=1,
            ),
        }
    else:
        error = pred.draws - obs_matrix
        metric_draws = {
            "rmse": np.sqrt(np.mean(error**2, axis=1)),
            "mae": np.mean(np.abs(error), axis=1),
        }
    long_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []
    for metric, values in metric_draws.items():
        long_rows.append(
            pd.DataFrame(
                {
                    "draw": np.arange(1, len(values) + 1, dtype=int),
                    "metric": metric,
                    "value": values,
                }
            )
        )
        q = np.quantile(values, qprobs, method="linear")
        summary_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "lower": float(q[0]),
                "median": float(q[1]),
                "upper": float(q[2]),
            }
        )
    return _PredictionScoreUncertainty(
        family=validated.family,
        scope="fitted_prepared_data" if newdata is None else "supplied_data",
        draws=pd.concat(long_rows, ignore_index=True),
        summary=pd.DataFrame(summary_rows),
        prediction=pred,
    )


def prediction_score_uncertainty_table(x: _PredictionScoreUncertainty) -> pd.DataFrame:
    if not isinstance(x, _PredictionScoreUncertainty):
        raise GP3BayesError("`x` must be gp3bayes prediction-score uncertainty.")
    return x.summary.copy()


def binary_calibration_uncertainty(
    fit: BinaryFit,
    newdata: pd.DataFrame | None = None,
    bins: int = 10,
    include_group_effects: bool = False,
    ndraws: int = 1000,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> _BinaryCalibrationUncertainty:
    validated = cast(BinaryFit, _validate_fit(fit, "binary"))
    bins_value = _positive_integer(bins, "bins")
    if bins_value < 2:
        raise GP3BayesError("`bins` must be one integer greater than or equal to 2.")
    qprobs = _probabilities(probs)
    prepared = cast(Any, validated.specification.prepared)
    data = prepared.data.copy() if newdata is None else newdata.copy()
    outcome = cast(str, validated.specification.contract.mappings["outcome"])
    if not isinstance(data, pd.DataFrame) or outcome not in data.columns:
        raise GP3BayesError("Calibration uncertainty requires observed binary outcomes.")
    observed = pd.to_numeric(data[outcome], errors="raise").to_numpy(dtype=float)
    if not np.isin(observed, (0.0, 1.0)).all():
        raise GP3BayesError("Calibration uncertainty requires observed binary outcomes.")
    pred = predict_model(
        validated,
        newdata=data,
        type="expected",
        include_group_effects=include_group_effects,
        allow_new_levels=False,
        ndraws=ndraws,
    )
    mean_p = np.mean(pred.draws, axis=0)
    breaks = np.linspace(0, 1, bins_value + 1)
    bin_codes = np.digitize(mean_p, breaks[1:-1], right=False) + 1
    rows: list[dict[str, float | int]] = []
    for code in sorted(set(bin_codes.tolist())):
        idx = np.flatnonzero(bin_codes == code)
        posterior_bin = np.mean(pred.draws[:, idx], axis=1)
        q = np.quantile(posterior_bin, qprobs, method="linear")
        rows.append(
            {
                "bin": int(code),
                "n": int(idx.size),
                "probability_lower_bound": float(breaks[code - 1]),
                "probability_upper_bound": float(breaks[code]),
                "observed_rate": float(np.mean(observed[idx])),
                "predicted_mean": float(np.mean(posterior_bin)),
                "predicted_lower": float(q[0]),
                "predicted_median": float(q[1]),
                "predicted_upper": float(q[2]),
            }
        )
    return _BinaryCalibrationUncertainty(
        table=pd.DataFrame(rows),
        bins_requested=bins_value,
        prediction=pred,
        scope="fitted_prepared_data" if newdata is None else "supplied_data",
    )


def binary_calibration_uncertainty_table(x: _BinaryCalibrationUncertainty) -> pd.DataFrame:
    if not isinstance(x, _BinaryCalibrationUncertainty):
        raise GP3BayesError("`x` must be gp3bayes binary calibration uncertainty.")
    return x.table.copy()


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise GP3BayesError("Matplotlib is required for plotting.") from exc
    return plt


def _figure_axis(title: str, xlabel: str, ylabel: str):
    plt = _plt()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_prediction_profile(x: _PredictionProfile):
    d = prediction_profile_table(x)
    fig, ax = _figure_axis("Model-based prediction profile", x.variable, "Posterior prediction")
    xv = d["profile_x"].to_numpy(dtype=float)
    ax.fill_between(xv, d["lower"], d["upper"], alpha=0.2)
    ax.plot(xv, d["predicted_median"])
    return fig


def plot_prediction_gradient(x: _PredictionProfile | pd.DataFrame):
    d = prediction_gradient_table(x) if isinstance(x, _PredictionProfile) else x
    required = {"gradient_midpoint", "gradient_median", "gradient_lower", "gradient_upper"}
    if not isinstance(d, pd.DataFrame) or not required.issubset(d.columns):
        raise GP3BayesError("`x` does not contain prediction-gradient summaries.")
    fig, ax = _figure_axis(
        "Prediction-profile gradient", "Predictor midpoint", "Finite-difference predictive gradient"
    )
    xv = d["gradient_midpoint"].to_numpy(dtype=float)
    ax.axhline(0, linestyle="--")
    ax.fill_between(xv, d["gradient_lower"], d["gradient_upper"], alpha=0.2)
    ax.plot(xv, d["gradient_median"])
    return fig


def _surface_plot(x: _PredictionSurface, value: str, title: str, label: str):
    d = prediction_surface_table(x)
    fig, ax = _figure_axis(title, x.x, x.y)
    pivot = d.pivot(index="surface_y", columns="surface_x", values=value)
    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        origin="lower",
        aspect="auto",
        extent=(
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.min()),
            float(pivot.index.max()),
        ),
    )
    fig.colorbar(image, ax=ax, label=label)
    return fig


def plot_prediction_surface(x: _PredictionSurface):
    return _surface_plot(
        x, "predicted_median", "Model-based prediction surface", "Posterior median"
    )


def plot_prediction_surface_uncertainty(x: _PredictionSurface):
    return _surface_plot(
        x, "interval_width", "Prediction-surface posterior uncertainty", "Interval width"
    )


def plot_prediction_contrast_profile(x: _PredictionContrastProfile):
    d = prediction_contrast_profile_table(x)
    fig, ax = _figure_axis(
        f"Prediction contrast profile: {x.contrast_levels[1]} versus {x.contrast_levels[0]}",
        x.variable,
        x.measure,
    )
    xv = d["profile_x"].to_numpy(dtype=float)
    ax.axhline(0 if x.measure == "difference" else 1, linestyle="--")
    ax.fill_between(xv, d["contrast_lower"], d["contrast_upper"], alpha=0.2)
    ax.plot(xv, d["contrast_median"])
    return fig


def plot_predictive_atlas_statistics(x: _PredictiveDistributionAtlas):
    if not isinstance(x, _PredictiveDistributionAtlas):
        raise GP3BayesError("`x` must be a gp3bayes predictive distribution atlas.")
    plt = _plt()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, metric in zip(axes, ("mean", "sd", "median", "q10", "q90"), strict=True):
        ax.hist(x.draw_statistics[metric], bins=20, density=True, alpha=0.7)
        ax.axvline(x.observed_statistics[metric], linestyle="--")
        ax.set_title(metric)
    fig.suptitle("Posterior-predictive distribution atlas")
    fig.tight_layout()
    return fig


def plot_predictive_quantile_envelope(x: pd.DataFrame):
    required = {
        "probability",
        "observed_quantile",
        "predictive_median",
        "predictive_lower",
        "predictive_upper",
    }
    if not isinstance(x, pd.DataFrame) or not required.issubset(x.columns):
        raise GP3BayesError("`x` must be a predictive quantile-envelope table.")
    fig, ax = _figure_axis(
        "Posterior-predictive quantile envelope", "Quantile probability", "Outcome quantile"
    )
    p = x["probability"].to_numpy(dtype=float)
    ax.fill_between(p, x["predictive_lower"], x["predictive_upper"], alpha=0.2)
    ax.plot(p, x["predictive_median"])
    ax.scatter(p, x["observed_quantile"], marker="x")
    return fig


def plot_prediction_score_uncertainty(x: _PredictionScoreUncertainty):
    if not isinstance(x, _PredictionScoreUncertainty):
        raise GP3BayesError("`x` must be gp3bayes prediction-score uncertainty.")
    plt = _plt()
    metrics = list(pd.unique(x.draws["metric"]))
    fig, axes = plt.subplots(1, len(metrics), squeeze=False, figsize=(5 * len(metrics), 4))
    for ax, metric in zip(axes.ravel(), metrics, strict=True):
        values = x.draws.loc[x.draws["metric"] == metric, "value"].to_numpy(dtype=float)
        ax.hist(values, bins=20, density=True, alpha=0.7)
        ax.set_title(str(metric))
        ax.set_xlabel("Score")
    fig.suptitle("Posterior uncertainty in prediction scores")
    fig.tight_layout()
    return fig


def plot_binary_calibration_uncertainty(x: _BinaryCalibrationUncertainty | pd.DataFrame):
    d = (
        binary_calibration_uncertainty_table(x)
        if isinstance(x, _BinaryCalibrationUncertainty)
        else x
    )
    required = {"observed_rate", "predicted_median", "predicted_lower", "predicted_upper"}
    if not isinstance(d, pd.DataFrame) or not required.issubset(d.columns):
        raise GP3BayesError("`x` does not contain calibration-uncertainty summaries.")
    fig, ax = _figure_axis(
        "Binary calibration with posterior uncertainty",
        "Posterior predicted probability",
        "Observed event rate",
    )
    ax.plot([0, 1], [0, 1], linestyle="--")
    xmed = d["predicted_median"].to_numpy(dtype=float)
    y = d["observed_rate"].to_numpy(dtype=float)
    xerr = np.vstack(
        (
            xmed - d["predicted_lower"].to_numpy(dtype=float),
            d["predicted_upper"].to_numpy(dtype=float) - xmed,
        )
    )
    ax.errorbar(xmed, y, xerr=xerr, fmt="o")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_prediction_draws(
    x: Prediction,
    observations: Sequence[int] | None = None,
    max_draws: int = 500,
):
    d = prediction_draws_long(x, max_draws=max_draws)
    if observations is not None:
        try:
            keep = {int(v) for v in observations}
        except (TypeError, ValueError) as exc:
            raise GP3BayesError("`observations` must be numeric prediction-row indices.") from exc
        d = d[d["observation"].isin(keep)]
    if d.empty:
        raise GP3BayesError("No prediction draws remain for plotting.")
    fig, ax = _figure_axis(
        "Posterior prediction distributions", "Prediction row", "Posterior predicted value"
    )
    groups = [
        group["value"].to_numpy(dtype=float) for _, group in d.groupby("observation", sort=True)
    ]
    labels = [str(v) for v in sorted(d["observation"].unique())]
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    return fig


def plot_ppc_statistic(x: _PosteriorPredictiveStatistic, bins: int = 30):
    if not isinstance(x, _PosteriorPredictiveStatistic):
        raise GP3BayesError("`x` must be a `gp3bayes_ppc_statistic`.")
    bins_value = _positive_integer(bins, "bins")
    if bins_value < 2:
        raise GP3BayesError("`bins` must be one integer greater than or equal to 2.")
    fig, ax = _figure_axis(
        "Posterior predictive discrepancy distribution",
        f"Replicated {x.statistic}",
        "Posterior predictive draws",
    )
    ax.hist(x.replicated, bins=bins_value)
    ax.axvline(x.observed, linestyle="--")
    return fig


def plot_binary_roc(x: object, observed: Sequence[float] | None = None):
    d = (
        x
        if isinstance(x, pd.DataFrame)
        and {"false_positive_rate", "true_positive_rate"}.issubset(x.columns)
        else binary_roc_curve(cast(Any, x), observed)
    )
    fig, ax = _figure_axis("Binary ROC curve", "False-positive rate", "True-positive rate")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(d["false_positive_rate"], d["true_positive_rate"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_binary_precision_recall(x: object, observed: Sequence[float] | None = None):
    d = (
        x
        if isinstance(x, pd.DataFrame) and {"recall", "precision"}.issubset(x.columns)
        else binary_precision_recall_curve(cast(Any, x), observed)
    )
    fig, ax = _figure_axis("Binary precision-recall curve", "Recall", "Precision")
    ax.plot(d["recall"], d["precision"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_binary_group_calibration(x: object, group: str | None = None):
    if isinstance(x, Prediction):
        if group is None:
            raise GP3BayesError("Supply `group` for a prediction object.")
        d = binary_group_calibration(x, group)
    else:
        d = x  # type: ignore[assignment]
    required = {"group", "predicted_probability", "observed_rate"}
    if not isinstance(d, pd.DataFrame) or not required.issubset(d.columns):
        raise GP3BayesError("`x` does not contain grouped calibration data.")
    fig, ax = _figure_axis(
        "Grouped binary calibration", "Mean predicted probability", "Observed rate"
    )
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.scatter(d["predicted_probability"], d["observed_rate"])
    for _, row in d.iterrows():
        ax.annotate(
            str(row["group"]), (float(row["predicted_probability"]), float(row["observed_rate"]))
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_duration_qq(x: Prediction | pd.DataFrame):
    d = duration_qq_table(x) if isinstance(x, Prediction) else x
    required = {
        "observed_quantile",
        "predictive_mean_quantile",
        "predictive_lower_quantile",
        "predictive_upper_quantile",
    }
    if not isinstance(d, pd.DataFrame) or not required.issubset(d.columns):
        raise GP3BayesError("`x` does not contain duration Q-Q data.")
    fig, ax = _figure_axis(
        "Duration posterior predictive Q-Q check",
        "Observed quantile",
        "Posterior predictive quantile",
    )
    lo = min(float(d["observed_quantile"].min()), float(d["predictive_lower_quantile"].min()))
    hi = max(float(d["observed_quantile"].max()), float(d["predictive_upper_quantile"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    y = d["predictive_mean_quantile"].to_numpy(dtype=float)
    yerr = np.vstack(
        (
            y - d["predictive_lower_quantile"].to_numpy(dtype=float),
            d["predictive_upper_quantile"].to_numpy(dtype=float) - y,
        )
    )
    ax.errorbar(d["observed_quantile"], y, yerr=yerr, fmt="o")
    return fig


def plot_duration_tail(x: pd.DataFrame):
    required = {
        "threshold",
        "observed_tail_rate",
        "predictive_mean_tail_rate",
        "predictive_lower_tail_rate",
        "predictive_upper_tail_rate",
    }
    if not isinstance(x, pd.DataFrame) or not required.issubset(x.columns):
        raise GP3BayesError("`x` must be a duration tail-check table.")
    fig, ax = _figure_axis(
        "Duration posterior predictive tail check", "Duration threshold", "Tail rate"
    )
    y = x["predictive_mean_tail_rate"].to_numpy(dtype=float)
    yerr = np.vstack(
        (
            y - x["predictive_lower_tail_rate"].to_numpy(dtype=float),
            x["predictive_upper_tail_rate"].to_numpy(dtype=float) - y,
        )
    )
    ax.errorbar(x["threshold"], y, yerr=yerr, fmt="o")
    ax.scatter(x["threshold"], x["observed_tail_rate"], marker="x")
    return fig


def plot_group_predictions(x: pd.DataFrame, group_column: str):
    required = {group_column, "predicted_median", "lower", "upper"}
    if not isinstance(x, pd.DataFrame) or not required.issubset(x.columns):
        raise GP3BayesError("`x` does not contain the requested group prediction summary.")
    fig, ax = _figure_axis(
        "Grouped posterior predictions", "Posterior group prediction", group_column
    )
    positions = np.arange(len(x))
    med = x["predicted_median"].to_numpy(dtype=float)
    err = np.vstack(
        (med - x["lower"].to_numpy(dtype=float), x["upper"].to_numpy(dtype=float) - med)
    )
    ax.errorbar(med, positions, xerr=err, fmt="o")
    if "observed" in x.columns:
        observed_values = pd.to_numeric(x["observed"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(observed_values)
        ax.scatter(observed_values[finite], positions[finite], marker="x")
    ax.set_yticks(positions, labels=x[group_column].astype(str))
    return fig


def plot_prediction_interval_width(x: Prediction | pd.DataFrame):
    d = prediction_interval_width(x) if isinstance(x, Prediction) else x
    if not isinstance(d, pd.DataFrame) or not {"observation", "interval_width"}.issubset(d.columns):
        raise GP3BayesError("`x` does not contain prediction interval-width data.")
    fig, ax = _figure_axis(
        "Posterior prediction interval widths", "Prediction row", "Interval width"
    )
    ax.plot(d["observation"], d["interval_width"], marker="o")
    return fig


def plot_prediction_rank_probabilities(x: pd.DataFrame):
    if not isinstance(x, pd.DataFrame) or not {"observation", "probability_rank_1"}.issubset(
        x.columns
    ):
        raise GP3BayesError("`x` must be a prediction ranking-probability table.")
    fig, ax = _figure_axis(
        "Descriptive posterior ranking probabilities",
        "Prediction row",
        "Posterior probability of rank 1",
    )
    ax.bar(x["observation"].astype(str), x["probability_rank_1"])
    ax.set_ylim(0, 1)
    return fig
