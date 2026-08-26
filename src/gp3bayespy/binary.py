"""Hierarchical binary workflow foundation.

Python-native port of the frozen gp3bayes 0.5.0
``binary-workflow-foundation.R`` layer.  The workflow is deliberately
backend-independent: simulation, deterministic preparation, prior declaration,
and prior-predictive simulation do not fit a Bayesian model.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.special import expit  # type: ignore[import-untyped]

from .contracts import ModelContract
from .exceptions import GP3BayesError
from .fitting import (
    _backend_versions,
    _load_pymc,
    _pymc_available,
    _require_pymc,
    _translation_parameter_table,
    _validate_sampling_controls,
)
from .readiness import ReadinessAudit, audit_model_readiness
from .specification import (
    ModelSpecification,
    PriorSpecification,
    create_model_specification,
    create_prior_specification,
    validate_prior_specification,
)

_DEFAULT_INTERCEPT = math.log(0.35 / 0.65)


def _numeric_scalar(
    value: object,
    name: str,
    *,
    lower: float = -math.inf,
    upper: float = math.inf,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be one finite numeric value.")
    number = float(value)
    lower_ok = number > lower if lower_open else number >= lower
    upper_ok = number < upper if upper_open else number <= upper
    if not lower_ok or not upper_ok:
        left = "(" if lower_open else "["
        right = ")" if upper_open else "]"
        raise GP3BayesError(
            f"`{name}` must lie in {left}{lower:g}, {upper:g}{right}."
        )
    return number


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    number = _numeric_scalar(value, name, lower=float(minimum))
    if number != math.floor(number):
        raise GP3BayesError(f"`{name}` must be integer-valued.")
    return int(number)


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise GP3BayesError(f"`{name}` must be TRUE or FALSE.")
    return value


def _character_vector(value: Sequence[str] | str | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    if (
        any(not isinstance(item, str) or not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise GP3BayesError(
            f"`{name}` must be a character vector of unique non-empty values."
        )
    return values


def _validate_binary_contract(contract: ModelContract) -> ModelContract:
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")
    if contract.family != "binary":
        raise GP3BayesError("This workflow requires a binary model contract.")
    return contract


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1))


def _standardize(values: np.ndarray) -> np.ndarray:
    center = float(np.mean(values))
    scale = _sample_sd(values)
    return (values - center) / scale


def _quote_name(value: str) -> str:
    if not value:
        raise GP3BayesError("Formula column names must be non-empty character scalars.")
    reserved = {
        "if",
        "else",
        "repeat",
        "while",
        "function",
        "for",
        "in",
        "next",
        "break",
        "TRUE",
        "FALSE",
        "NULL",
        "Inf",
        "NaN",
        "NA",
        "NA_integer_",
        "NA_real_",
        "NA_complex_",
        "NA_character_",
    }
    syntactic = (
        value not in reserved
        and (
            value[0].isalpha()
            or (value[0] == "." and (len(value) == 1 or not value[1].isdigit()))
        )
        and all(char.isalnum() or char in "._" for char in value[1:])
    )
    if syntactic:
        return value
    escaped = value.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def _fixed_formula_text(contract: ModelContract) -> str:
    _validate_binary_contract(contract)
    mappings = contract.mappings
    terms: list[str] = []
    for value in (mappings["condition"], mappings["time"], *contract.predictors):
        if value is not None and value not in terms:
            terms.append(value)
    quoted = [_quote_name(value) for value in terms]
    if contract.interaction is not None:
        interaction = ":".join(_quote_name(value) for value in contract.interaction)
        if interaction not in quoted:
            quoted.append(interaction)
    rhs = " + ".join(quoted) if quoted else "1"
    return f"{_quote_name(cast(str, mappings['outcome']))} ~ {rhs}"


def _column_basis(series: pd.Series, name: str) -> list[tuple[str, np.ndarray]]:
    if pd.api.types.is_bool_dtype(series.dtype):
        return [(name, series.astype(float).to_numpy())]
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
        return [(name, values)]

    if isinstance(series.dtype, pd.CategoricalDtype):
        levels = [str(value) for value in series.cat.categories]
    else:
        levels = sorted(str(value) for value in pd.unique(series.astype(str)))
    if len(levels) < 2:
        return []
    text = series.astype(str)
    return [
        (f"{name}{level}", text.eq(level).to_numpy(dtype=float))
        for level in levels[1:]
    ]


def _fixed_model_matrix(
    data: pd.DataFrame, contract: ModelContract
) -> tuple[np.ndarray, tuple[str, ...]]:
    mappings = contract.mappings
    fixed_names: list[str] = []
    for value in (mappings["condition"], mappings["time"], *contract.predictors):
        if value is not None and value not in fixed_names:
            fixed_names.append(value)

    columns: list[np.ndarray] = [np.ones(len(data), dtype=float)]
    names: list[str] = ["(Intercept)"]
    bases: dict[str, list[tuple[str, np.ndarray]]] = {}
    for name in fixed_names:
        basis = _column_basis(data[name], name)
        bases[name] = basis
        for column_name, values in basis:
            columns.append(values)
            names.append(column_name)

    if contract.interaction is not None:
        left, right = contract.interaction
        left_basis = bases.get(left) or _column_basis(data[left], left)
        right_basis = bases.get(right) or _column_basis(data[right], right)
        for left_name, left_values in left_basis:
            for right_name, right_values in right_basis:
                columns.append(left_values * right_values)
                names.append(f"{left_name}:{right_name}")

    matrix = np.column_stack(columns)
    return matrix, tuple(names)


def _map_binary_outcome(
    series: pd.Series,
    mapping: Mapping[object, int] | None,
) -> tuple[pd.Series, Mapping[object, int]]:
    if pd.api.types.is_bool_dtype(series.dtype):
        result = series.astype(int)
        return result, {0: 0, 1: 1}

    if pd.api.types.is_numeric_dtype(series.dtype):
        observed = pd.to_numeric(series.dropna(), errors="coerce")
        if len(observed) > 0 and observed.isin([0, 1]).all():
            return series.astype(int), {0: 0, 1: 1}

    if mapping is None:
        raise GP3BayesError(
            "A named `outcome_mapping` with values 0 and 1 is required for labelled "
            "binary outcomes."
        )
    if len(mapping) != 2 or set(mapping.values()) != {0, 1}:
        raise GP3BayesError(
            "`outcome_mapping` must be a named vector mapping exactly two distinct labels "
            "to 0 and 1."
        )
    labels = series.astype(str)
    observed_labels = list(dict.fromkeys(labels.tolist()))
    missing_labels = [label for label in observed_labels if label not in mapping]
    if missing_labels:
        raise GP3BayesError(
            "`outcome_mapping` does not cover observed labels: "
            + ", ".join(str(value) for value in missing_labels)
            + "."
        )
    mapped = labels.map(mapping)
    return mapped.astype(int), dict(mapping)


def _code_condition(
    series: pd.Series,
    condition_levels: Sequence[object] | None,
    condition_coding: Sequence[float],
) -> tuple[pd.Series, tuple[str, str], dict[str, float]]:
    coding = tuple(float(value) for value in condition_coding)
    if (
        len(coding) != 2
        or not all(math.isfinite(value) for value in coding)
        or coding[0] == coding[1]
    ):
        raise GP3BayesError(
            "`condition_coding` must contain two distinct finite numeric values."
        )

    observed_raw = list(dict.fromkeys(series.dropna().tolist()))
    if len(observed_raw) != 2:
        raise GP3BayesError(
            "The focal condition must have exactly two observed non-missing levels."
        )
    observed_text = [str(value) for value in observed_raw]

    if condition_levels is None:
        if isinstance(series.dtype, pd.CategoricalDtype):
            levels = [
                str(level)
                for level in series.cat.categories
                if str(level) in set(observed_text)
            ]
        elif pd.api.types.is_numeric_dtype(series.dtype):
            levels = [str(value) for value in sorted(observed_raw)]
        else:
            levels = sorted(observed_text)
    else:
        levels = [str(value) for value in condition_levels]

    if len(levels) != 2 or set(levels) != set(observed_text):
        raise GP3BayesError(
            "`condition_levels` must list the two observed levels in reference-to-focal order."
        )

    mapping = {levels[0]: coding[0], levels[1]: coding[1]}
    values = series.astype(str).map(mapping)
    return values.astype(float), (levels[0], levels[1]), mapping


@dataclass(frozen=True, slots=True)
class BinarySimulation:
    """Synthetic hierarchical Bernoulli-logit data and generating truth."""

    simulation_version: str
    data: pd.DataFrame
    truth: Mapping[str, Any]
    random_effects: Mapping[str, pd.DataFrame | None]
    design: Mapping[str, Any]

    def __repr__(self) -> str:
        fixed = cast(Mapping[str, float], self.truth["fixed_effects"])
        return "\n".join(
            [
                "<gp3bayes_binary_simulation>",
                f"  Rows: {len(self.data)}",
                f"  Participants: {self.design['n_participants']}",
                f"  Items: {self.design['n_items']}",
                f"  True condition effect: {fixed['condition']:g}",
                f"  Seed: {self.truth['seed']}",
            ]
        )


def simulate_hierarchical_binary_data(
    n_participants: int = 40,
    trials_per_participant: int = 20,
    n_items: int = 20,
    intercept: float = _DEFAULT_INTERCEPT,
    condition_effect: float = 0.8,
    participant_covariate_effect: float = 0.3,
    trial_covariate_effect: float = 0.15,
    interaction_effect: float = 0.25,
    participant_sd: float = 0.7,
    item_sd: float = 0.35,
    random_slope_sd: float = 0.3,
    random_slope_cor: float = 0.0,
    condition_probability: float = 0.5,
    balanced_condition: bool = True,
    include_items: bool = True,
    seed: int = 1,
) -> BinarySimulation:
    """Simulate governed repeated-measures binary data without model fitting."""
    n_participants = _integer(n_participants, "n_participants", minimum=2)
    trials_per_participant = _integer(
        trials_per_participant, "trials_per_participant", minimum=2
    )
    n_items = _integer(n_items, "n_items", minimum=2)
    intercept = _numeric_scalar(intercept, "intercept")
    condition_effect = _numeric_scalar(condition_effect, "condition_effect")
    participant_covariate_effect = _numeric_scalar(
        participant_covariate_effect, "participant_covariate_effect"
    )
    trial_covariate_effect = _numeric_scalar(trial_covariate_effect, "trial_covariate_effect")
    interaction_effect = _numeric_scalar(interaction_effect, "interaction_effect")
    participant_sd = _numeric_scalar(participant_sd, "participant_sd", lower=0)
    item_sd = _numeric_scalar(item_sd, "item_sd", lower=0)
    random_slope_sd = _numeric_scalar(random_slope_sd, "random_slope_sd", lower=0)
    random_slope_cor = _numeric_scalar(
        random_slope_cor,
        "random_slope_cor",
        lower=-1,
        upper=1,
        lower_open=True,
        upper_open=True,
    )
    condition_probability = _numeric_scalar(
        condition_probability,
        "condition_probability",
        lower=0,
        upper=1,
        lower_open=True,
        upper_open=True,
    )
    balanced_condition = _flag(balanced_condition, "balanced_condition")
    include_items = _flag(include_items, "include_items")
    seed = _integer(seed, "seed", minimum=0)

    rng = np.random.RandomState(seed)
    participant_levels = np.array(
        [f"p{index:03d}" for index in range(1, n_participants + 1)], dtype=object
    )
    participant_id = np.repeat(participant_levels, trials_per_participant)
    trial_id = np.tile(np.arange(1, trials_per_participant + 1), n_participants)
    n_rows = len(participant_id)

    if balanced_condition:
        sequences: list[np.ndarray] = []
        base = np.resize(np.array([-0.5, 0.5], dtype=float), trials_per_participant)
        for _ in range(n_participants):
            sequences.append(rng.permutation(base))
        condition_code = np.concatenate(sequences)
    else:
        condition_code = np.where(rng.uniform(size=n_rows) < condition_probability, 0.5, -0.5)

    participant_covariate_by_id = _standardize(rng.normal(size=n_participants))
    participant_index = pd.Categorical(participant_id, categories=participant_levels).codes
    participant_covariate = participant_covariate_by_id[participant_index]
    trial_covariate = _standardize(rng.normal(size=n_rows))

    z_intercept = rng.normal(size=n_participants)
    z_slope = rng.normal(size=n_participants)
    participant_intercept = participant_sd * z_intercept
    participant_slope = random_slope_sd * (
        random_slope_cor * z_intercept
        + math.sqrt(1 - random_slope_cor**2) * z_slope
    )

    item_id: np.ndarray | None = None
    item_effect = np.zeros(n_rows, dtype=float)
    item_effect_by_id = np.array([], dtype=float)
    item_levels = np.array([], dtype=object)
    if include_items:
        item_levels = np.array([f"i{index:03d}" for index in range(1, n_items + 1)], dtype=object)
        participant_offsets = np.repeat(np.arange(n_participants), trials_per_participant)
        item_index = (trial_id + participant_offsets - 1) % n_items
        item_id = item_levels[item_index]
        item_effect_by_id = rng.normal(loc=0.0, scale=item_sd, size=n_items)
        item_effect = item_effect_by_id[item_index]

    linear_predictor = (
        intercept
        + condition_effect * condition_code
        + participant_covariate_effect * participant_covariate
        + trial_covariate_effect * trial_covariate
        + interaction_effect * condition_code * participant_covariate
        + participant_intercept[participant_index]
        + participant_slope[participant_index] * condition_code
        + item_effect
    )
    probability = expit(linear_predictor)
    selected = rng.binomial(1, probability, size=n_rows).astype(int)

    data_dict: dict[str, Any] = {
        "participant_id": participant_id,
        "trial_id": trial_id.astype(int),
        "condition": pd.Categorical(
            np.where(condition_code < 0, "control", "treatment"),
            categories=["control", "treatment"],
            ordered=False,
        ),
        "participant_covariate": participant_covariate,
        "trial_covariate": trial_covariate,
        "selected": selected,
        "true_probability": probability,
    }
    if include_items:
        data_dict["item_id"] = item_id
    preferred = [
        "participant_id",
        *( ["item_id"] if include_items else [] ),
        "trial_id",
        "condition",
        "participant_covariate",
        "trial_covariate",
        "selected",
        "true_probability",
    ]
    data = pd.DataFrame(data_dict).loc[:, preferred]

    fixed_effects = {
        "(Intercept)": intercept,
        "condition": condition_effect,
        "participant_covariate": participant_covariate_effect,
        "trial_covariate": trial_covariate_effect,
        "condition:participant_covariate": interaction_effect,
    }
    truth: dict[str, Any] = {
        "fixed_effects": fixed_effects,
        "participant_sd": participant_sd,
        "item_sd": item_sd if include_items else 0.0,
        "random_slope_sd": random_slope_sd,
        "random_slope_cor": random_slope_cor,
        "baseline_probability": float(expit(intercept)),
        "condition_coding": {"control": -0.5, "treatment": 0.5},
        "condition_probability": condition_probability,
        "balanced_condition": balanced_condition,
        "seed": seed,
    }
    participant_re = pd.DataFrame(
        {
            "participant_id": participant_levels,
            "intercept": participant_intercept,
            "condition_slope": participant_slope,
            "participant_covariate": participant_covariate_by_id,
        }
    )
    item_re = (
        pd.DataFrame({"item_id": item_levels, "intercept": item_effect_by_id})
        if include_items
        else None
    )
    design = {
        "n_participants": n_participants,
        "trials_per_participant": trials_per_participant,
        "n_items": n_items if include_items else 0,
        "include_items": include_items,
        "random_slope": random_slope_sd > 0,
        "row_count": n_rows,
    }
    return BinarySimulation(
        simulation_version="0.1",
        data=data,
        truth=truth,
        random_effects={"participant": participant_re, "item": item_re},
        design=design,
    )


@dataclass(frozen=True, slots=True)
class BinaryPrepared:
    """Deterministically prepared binary analysis data."""

    preparation_version: str
    data: pd.DataFrame
    contract: ModelContract
    audit: ReadinessAudit
    transformations: Mapping[str, Any]
    decision_log: pd.DataFrame
    fixed_formula: str
    fixed_formula_text: str
    model_matrix_columns: tuple[str, ...]
    n_input_rows: int
    n_analysis_rows: int
    rows_removed: int
    contains_data: bool = True
    backend: str = "none"
    fit_performed: bool = False

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_binary_prepared>",
                f"  Input rows: {self.n_input_rows}",
                f"  Analysis rows: {self.n_analysis_rows}",
                f"  Rows removed: {self.rows_removed}",
                f"  Readiness: {self.audit.status}",
                "  Fixed matrix columns: " + ", ".join(self.model_matrix_columns),
                "  Backend: none",
                "  Fit performed: FALSE",
            ]
        )


def prepare_hierarchical_binary_data(
    data: pd.DataFrame,
    contract: ModelContract,
    outcome_mapping: Mapping[object, int] | None = None,
    condition_levels: Sequence[object] | None = None,
    condition_coding: Sequence[float] = (-0.5, 0.5),
    scale_predictors: Sequence[str] | str = (),
    scale_time: bool = False,
    missing: str = "error",
) -> BinaryPrepared:
    """Apply explicit binary mappings, condition coding, scaling, and readiness gating."""
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    _validate_binary_contract(contract)
    if missing not in {"error", "drop"}:
        raise GP3BayesError('`missing` must be either "error" or "drop".')
    scale_predictors_tuple = _character_vector(scale_predictors, "scale_predictors")
    scale_time = _flag(scale_time, "scale_time")
    undeclared = [value for value in scale_predictors_tuple if value not in contract.predictors]
    if undeclared:
        raise GP3BayesError(
            "Every `scale_predictors` entry must be declared in `contract$predictors`. "
            "Undeclared: "
            + ", ".join(undeclared)
            + "."
        )

    required: list[str] = []
    for value in (*contract.mappings.values(), *contract.predictors, *(contract.interaction or ())):
        if value is not None and value not in required:
            required.append(value)
    absent = [value for value in required if value not in data.columns]
    if absent:
        raise GP3BayesError("Required columns are missing: " + ", ".join(absent) + ".")

    required_frame = cast(pd.DataFrame, data.loc[:, required])
    complete_rows = ~required_frame.isna().any(axis=1)
    dropped_positions = [index + 1 for index, complete in enumerate(complete_rows) if not complete]
    if dropped_positions and missing == "error":
        raise GP3BayesError(
            "Missing values are present in declared analysis columns. Use `missing = \"drop\"` "
            "only after an explicit decision."
        )
    working = (
        cast(pd.DataFrame, data.loc[complete_rows]).copy()
        if dropped_positions
        else data.copy()
    )
    if working.empty:
        raise GP3BayesError("No complete analysis rows remain.")

    outcome_column = cast(str, contract.mappings["outcome"])
    original_outcome_type = str(working[outcome_column].dtype)
    mapped_outcome, stored_mapping = _map_binary_outcome(working[outcome_column], outcome_mapping)
    working[outcome_column] = mapped_outcome

    transformations: dict[str, Any] = {
        "outcome": {
            "column": outcome_column,
            "original_class": original_outcome_type,
            "mapping": stored_mapping,
        },
        "condition": None,
        "numeric_scaling": {},
        "missing": {
            "action": missing,
            "dropped_row_positions": tuple(dropped_positions),
            "retained_row_positions": tuple(
                index + 1 for index, complete in enumerate(complete_rows) if complete
            ),
        },
    }

    condition_column = contract.mappings["condition"]
    if condition_column is not None:
        coded, source_levels, coding = _code_condition(
            working[condition_column], condition_levels, condition_coding
        )
        working[condition_column] = coded
        transformations["condition"] = {
            "column": condition_column,
            "source_levels": source_levels,
            "coding": coding,
        }

    columns_to_scale = list(scale_predictors_tuple)
    time_column = contract.mappings["time"]
    if scale_time and time_column is not None and time_column not in columns_to_scale:
        columns_to_scale.append(time_column)
    numeric_scaling = cast(dict[str, dict[str, float]], transformations["numeric_scaling"])
    for column in columns_to_scale:
        series = working[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            raise GP3BayesError(f"Scaled column `{column}` must be finite and numeric.")
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise GP3BayesError(f"Scaled column `{column}` must be finite and numeric.")
        center = float(np.mean(values))
        scale_value = _sample_sd(values)
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise GP3BayesError(f"Scaled column `{column}` must have positive variation.")
        working[column] = (values - center) / scale_value
        numeric_scaling[column] = {"center": center, "scale": scale_value}

    audit = audit_model_readiness(working, contract)
    if not audit.ready:
        raise GP3BayesError(
            f"Prepared data failed the model-readiness gate with status `{audit.status}`."
        )

    fixed_formula = _fixed_formula_text(contract)
    model_matrix, matrix_columns = _fixed_model_matrix(working, contract)
    if np.linalg.matrix_rank(model_matrix) < model_matrix.shape[1]:
        raise GP3BayesError("The prepared fixed-effects design matrix is rank deficient.")

    outcome_map = cast(Mapping[object, int], transformations["outcome"]["mapping"])
    outcome_value = "; ".join(f"{key}={value}" for key, value in outcome_map.items())
    condition_transform = transformations["condition"]
    if condition_transform is None:
        condition_value = "not_applicable"
    else:
        condition_map = cast(Mapping[str, float], condition_transform["coding"])
        condition_value = "; ".join(f"{key}={value:g}" for key, value in condition_map.items())
    scaling_value = ", ".join(numeric_scaling) if numeric_scaling else "none"
    decision_log = pd.DataFrame(
        {
            "decision": [
                "binary_outcome_mapping",
                "condition_coding",
                "numeric_scaling",
                "missing_rows",
            ],
            "value": [
                outcome_value,
                condition_value,
                scaling_value,
                f"{missing}; dropped={len(dropped_positions)}",
            ],
        }
    )

    return BinaryPrepared(
        preparation_version="0.1",
        data=working,
        contract=contract,
        audit=audit,
        transformations=transformations,
        decision_log=decision_log,
        fixed_formula=fixed_formula,
        fixed_formula_text=fixed_formula,
        model_matrix_columns=matrix_columns,
        n_input_rows=len(data),
        n_analysis_rows=len(working),
        rows_removed=len(dropped_positions),
    )


@dataclass(frozen=True, slots=True)
class BinaryModelSpecification(ModelSpecification):
    """Backend-independent binary model specification."""

    binary_workflow_version: str = "0.1"
    prepared: BinaryPrepared | None = None
    fixed_formula: str = ""
    fixed_formula_text: str = ""
    model_matrix_columns: tuple[str, ...] = ()
    contains_data: bool = True
    fitting_engine: str = "none"
    backend_dependency: str = "none"
    unrestricted_formula: bool = False
    prior_predictive_performed: bool = False

    def __repr__(self) -> str:
        baseline = self.priors.baseline
        return "\n".join(
            [
                "<gp3bayes_binary_model_specification>",
                f"  Formula: {self.formula_text}",
                f"  Fixed formula: {self.fixed_formula_text}",
                f"  Baseline probability: {baseline:g}",
                f"  Readiness: {self.readiness_status}",
                "  Fitting engine: none",
                "  Backend dependency: none",
                "  Fit performed: FALSE",
            ]
        )


def specify_binary_model(
    prepared: BinaryPrepared,
    baseline: float = 0.5,
    intercept_scale: float = 1.5,
    coefficient_scale: float = 0.75,
    group_sd_scale: float = 1.0,
    correlation_eta: float = 2.0,
    student_df: float = 3.0,
) -> BinaryModelSpecification:
    """Combine prepared data, validated priors, and the restricted model formula."""
    if not isinstance(prepared, BinaryPrepared):
        raise GP3BayesError("`prepared` must inherit from `gp3bayes_binary_prepared`.")
    _validate_binary_contract(prepared.contract)
    if not prepared.audit.ready:
        raise GP3BayesError("`prepared$audit` is not ready for specification.")

    priors = create_prior_specification(
        prepared.contract,
        baseline=baseline,
        intercept_scale=intercept_scale,
        coefficient_scale=coefficient_scale,
        group_sd_scale=group_sd_scale,
        correlation_eta=correlation_eta,
        student_df=student_df,
    )
    core = create_model_specification(prepared.contract, prepared.audit, priors)
    core_values = {field.name: getattr(core, field.name) for field in fields(ModelSpecification)}
    return BinaryModelSpecification(
        **core_values,
        binary_workflow_version="0.1",
        prepared=prepared,
        fixed_formula=prepared.fixed_formula,
        fixed_formula_text=prepared.fixed_formula_text,
        model_matrix_columns=prepared.model_matrix_columns,
    )


def _prior_row(priors: PriorSpecification, parameter_class: str) -> pd.Series:
    validate_prior_specification(priors)
    mask = priors.table["parameter_class"].eq(parameter_class)
    if int(mask.sum()) != 1:
        raise GP3BayesError(
            f"The prior specification must contain exactly one `{parameter_class}` row."
        )
    return cast(pd.Series, priors.table.loc[mask].iloc[0])


def _binary_summary(
    y: np.ndarray,
    probability: np.ndarray,
    condition: np.ndarray | None,
    participant: np.ndarray,
    item: np.ndarray | None,
    boundary_probability: tuple[float, float],
) -> dict[str, float]:
    participant_levels = pd.unique(participant)
    participant_rates = np.array(
        [float(np.mean(y[participant == level])) for level in participant_levels]
    )
    condition_low_rate = math.nan
    condition_high_rate = math.nan
    condition_rate_contrast = math.nan
    if condition is not None:
        levels = np.sort(np.unique(condition))
        if len(levels) == 2:
            rates = [float(np.mean(y[condition == level])) for level in levels]
            condition_low_rate, condition_high_rate = rates
            condition_rate_contrast = rates[1] - rates[0]
    item_rate_sd = math.nan
    if item is not None:
        item_levels = pd.unique(item)
        item_rates = np.array([float(np.mean(y[item == level])) for level in item_levels])
        if len(item_rates) > 1:
            item_rate_sd = _sample_sd(item_rates)
    return {
        "overall_rate": float(np.mean(y)),
        "condition_low_rate": condition_low_rate,
        "condition_high_rate": condition_high_rate,
        "condition_rate_contrast": condition_rate_contrast,
        "participant_rate_sd": (
            _sample_sd(participant_rates) if len(participant_rates) > 1 else math.nan
        ),
        "item_rate_sd": item_rate_sd,
        "participant_all_zero": float(np.mean(participant_rates == 0)),
        "participant_all_one": float(np.mean(participant_rates == 1)),
        "probability_below_boundary": float(np.mean(probability < boundary_probability[0])),
        "probability_above_boundary": float(np.mean(probability > boundary_probability[1])),
    }


@dataclass(frozen=True, slots=True)
class BinaryPriorPredictiveCheck:
    """Backend-independent prior predictive audit for a binary specification."""

    check_version: str
    family: str
    adequate: bool
    draws: int
    summaries: pd.DataFrame
    checks: pd.DataFrame
    thresholds: Mapping[str, Any]
    seed: int
    backend: str = "none"
    fitting_performed: bool = False
    interpretation: str = (
        "Failure indicates that the declared priors generate outcomes requiring substantive "
        "review under the approved design; priors are not altered or selected automatically."
    )
    limitations: str = (
        "This is a prior predictive simulation, not evidence of posterior adequacy, convergence, "
        "causal identification, or substantive validity."
    )

    def __repr__(self) -> str:
        failures = int((self.checks["status"] == "fail").sum())
        return "\n".join(
            [
                "<gp3bayes_binary_prior_predictive_check>",
                f"  Draws: {self.draws}",
                f"  Adequate: {str(self.adequate).upper()}",
                f"  Failed checks: {failures}",
                "  Backend: none",
                "  Fit performed: FALSE",
            ]
        )


def _probability_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    pair = tuple(float(value) for value in values)
    if (
        len(pair) != 2
        or not all(math.isfinite(value) for value in pair)
        or pair[0] < 0
        or pair[1] > 1
        or pair[0] >= pair[1]
    ):
        raise GP3BayesError(f"`{name}` must be an increasing pair between zero and one.")
    return pair[0], pair[1]


def check_binary_prior_predictive(
    specification: BinaryModelSpecification,
    draws: int = 500,
    seed: int = 1,
    plausible_rate: Sequence[float] = (0.01, 0.99),
    boundary_probability: Sequence[float] = (0.01, 0.99),
    extreme_contrast: float = 0.8,
    maximum_degenerate_participant_fraction: float = 0.5,
    maximum_boundary_mass: float = 0.5,
    maximum_extreme_probability: float = 0.25,
) -> BinaryPriorPredictiveCheck:
    """Simulate prior predictive outcomes from the declared design without fitting."""
    if not isinstance(specification, BinaryModelSpecification):
        raise GP3BayesError(
            "`specification` must inherit from `gp3bayes_binary_model_specification`."
        )
    prepared = specification.prepared
    if not isinstance(prepared, BinaryPrepared):
        raise GP3BayesError(
            "`specification$prepared` must inherit from `gp3bayes_binary_prepared`."
        )
    validate_prior_specification(specification.priors, specification.contract)
    draws = _integer(draws, "draws", minimum=50)
    seed = _integer(seed, "seed", minimum=0)
    plausible = _probability_pair(plausible_rate, "plausible_rate")
    boundary = _probability_pair(boundary_probability, "boundary_probability")
    extreme_contrast = _numeric_scalar(extreme_contrast, "extreme_contrast", lower=0, upper=1)
    maximum_degenerate_participant_fraction = _numeric_scalar(
        maximum_degenerate_participant_fraction,
        "maximum_degenerate_participant_fraction",
        lower=0,
        upper=1,
    )
    maximum_boundary_mass = _numeric_scalar(
        maximum_boundary_mass, "maximum_boundary_mass", lower=0, upper=1
    )
    maximum_extreme_probability = _numeric_scalar(
        maximum_extreme_probability, "maximum_extreme_probability", lower=0, upper=1
    )

    data = prepared.data
    contract = specification.contract
    model_matrix, _ = _fixed_model_matrix(data, contract)
    participant_column = cast(str, contract.mappings["participant"])
    participant = data[participant_column].astype(str).to_numpy()
    _, participant_index = np.unique(participant, return_inverse=True)
    participant_count = int(participant_index.max()) + 1

    item: np.ndarray | None = None
    item_index: np.ndarray | None = None
    item_count = 0
    item_column = contract.mappings["item"]
    if item_column is not None:
        item = data[item_column].astype(str).to_numpy()
        _, item_index = np.unique(item, return_inverse=True)
        item_count = int(item_index.max()) + 1

    condition: np.ndarray | None = None
    condition_column = contract.mappings["condition"]
    if condition_column is not None:
        condition = pd.to_numeric(data[condition_column], errors="raise").to_numpy(dtype=float)

    intercept_prior = _prior_row(specification.priors, "Intercept")
    coefficient_prior = _prior_row(specification.priors, "b")
    group_sd_prior = _prior_row(specification.priors, "sd")
    correlation_prior = _prior_row(specification.priors, "cor") if contract.random_slope else None

    rng = np.random.RandomState(seed)
    rows: list[dict[str, float]] = []
    for _ in range(draws):
        coefficients: np.ndarray = np.asarray(
            rng.normal(
                loc=float(coefficient_prior["location"]),
                scale=float(coefficient_prior["scale"]),
                size=model_matrix.shape[1],
            ),
            dtype=float,
        )
        coefficients[0] = rng.normal(
            loc=float(intercept_prior["location"]),
            scale=float(intercept_prior["scale"]),
        )
        participant_intercept_sd = abs(rng.standard_t(float(group_sd_prior["df"]))) * float(
            group_sd_prior["scale"]
        )
        z_intercept = rng.normal(size=participant_count)
        participant_intercept = participant_intercept_sd * z_intercept
        linear_predictor = model_matrix @ coefficients + participant_intercept[participant_index]

        if contract.random_slope:
            assert condition is not None
            assert correlation_prior is not None
            participant_slope_sd = abs(rng.standard_t(float(group_sd_prior["df"]))) * float(
                group_sd_prior["scale"]
            )
            shape = float(correlation_prior["shape"])
            correlation = 2 * rng.beta(shape, shape) - 1
            z_slope = rng.normal(size=participant_count)
            participant_slope = participant_slope_sd * (
                correlation * z_intercept + math.sqrt(1 - correlation**2) * z_slope
            )
            linear_predictor = linear_predictor + participant_slope[participant_index] * condition

        if item is not None:
            assert item_index is not None
            item_sd = abs(rng.standard_t(float(group_sd_prior["df"]))) * float(
                group_sd_prior["scale"]
            )
            item_intercept = rng.normal(loc=0.0, scale=item_sd, size=item_count)
            linear_predictor = linear_predictor + item_intercept[item_index]

        probability = expit(linear_predictor)
        replicated = rng.binomial(1, probability, size=len(data)).astype(int)
        rows.append(
            _binary_summary(
                replicated,
                probability,
                condition,
                participant,
                item,
                boundary,
            )
        )

    summaries = pd.DataFrame(rows)
    overall_rate_probability = float(
        np.mean(
            (summaries["overall_rate"] < plausible[0])
            | (summaries["overall_rate"] > plausible[1])
        )
    )
    condition_available = condition is not None
    if condition_available:
        condition_rate_probability = float(
            np.mean(
                (summaries["condition_low_rate"] < plausible[0])
                | (summaries["condition_low_rate"] > plausible[1])
                | (summaries["condition_high_rate"] < plausible[0])
                | (summaries["condition_high_rate"] > plausible[1])
            )
        )
        condition_contrast_probability = float(
            np.mean(np.abs(summaries["condition_rate_contrast"]) > extreme_contrast)
        )
    else:
        condition_rate_probability = math.nan
        condition_contrast_probability = math.nan
    participant_degeneracy_probability = float(
        np.mean(
            summaries["participant_all_zero"] + summaries["participant_all_one"]
            > maximum_degenerate_participant_fraction
        )
    )
    boundary_mass_probability = float(
        np.mean(
            summaries["probability_below_boundary"] + summaries["probability_above_boundary"]
            > maximum_boundary_mass
        )
    )

    check_names = [
        "overall_rate",
        "condition_rates",
        "condition_contrast",
        "participant_degeneracy",
        "boundary_probability_mass",
    ]
    probabilities = [
        overall_rate_probability,
        condition_rate_probability,
        condition_contrast_probability,
        participant_degeneracy_probability,
        boundary_mass_probability,
    ]
    applicable = [True, condition_available, condition_available, True, True]
    statuses = [
        "not_applicable"
        if not applies
        else (
            "pass"
            if math.isfinite(probability)
            and probability <= maximum_extreme_probability
            else "fail"
        )
        for applies, probability in zip(applicable, probabilities, strict=True)
    ]
    checks = pd.DataFrame(
        {
            "check": check_names,
            "probability": probabilities,
            "threshold": [maximum_extreme_probability] * len(check_names),
            "status": statuses,
        }
    )
    adequate = all(status == "pass" for status in statuses if status != "not_applicable")
    return BinaryPriorPredictiveCheck(
        check_version="0.1",
        family="binary",
        adequate=adequate,
        draws=draws,
        summaries=summaries,
        checks=checks,
        thresholds={
            "plausible_rate": plausible,
            "boundary_probability": boundary,
            "extreme_contrast": extreme_contrast,
            "maximum_degenerate_participant_fraction": maximum_degenerate_participant_fraction,
            "maximum_boundary_mass": maximum_boundary_mass,
            "maximum_extreme_probability": maximum_extreme_probability,
        },
        seed=seed,
    )


def _validate_binary_model_specification(
    specification: BinaryModelSpecification,
) -> BinaryModelSpecification:
    if not isinstance(specification, BinaryModelSpecification):
        raise GP3BayesError(
            "`specification` must inherit from `gp3bayes_binary_model_specification`."
        )
    if specification.family != "binary":
        raise GP3BayesError("`specification` must use the approved binary family.")
    _validate_binary_contract(specification.contract)
    if not isinstance(specification.prepared, BinaryPrepared):
        raise GP3BayesError(
            "`specification$prepared` must inherit from `gp3bayes_binary_prepared`."
        )
    if not specification.audit.ready:
        raise GP3BayesError("`specification$audit` must pass the readiness gate.")
    if specification.contract.link != "logit":
        raise GP3BayesError("Binary model fitting requires the approved logit link.")
    if specification.contract.likelihood != "Bernoulli":
        raise GP3BayesError(
            "Binary model fitting requires the approved Bernoulli likelihood."
        )
    return specification


@dataclass(frozen=True, slots=True)
class BinaryBackendSpecification:
    """Restricted Python backend translation of the R brms specification contract."""

    translation_version: str
    family: str
    model_family: str
    formula: str
    formula_text: str
    family_object: Mapping[str, str]
    priors: Mapping[str, str]
    prior_text: Mapping[str, str]
    validated_priors: pd.DataFrame
    parameter_table: pd.DataFrame
    specification: BinaryModelSpecification
    backend_interface: str
    sampling_backend: str
    algorithm: str
    backend_available: bool
    unrestricted_formula: bool = False
    compiled: bool = False
    fit_performed: bool = False
    diagnostics_assessed: bool = False
    source_backend_interface: str = "brms"
    source_sampling_backend: str = "rstan"
    source_algorithm: str = "sampling"
    intentional_python_divergence: str = (
        "The frozen R implementation translates to brms/rstan. The Python port preserves "
        "the restricted formula/prior contract and executes with optional PyMC NUTS."
    )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_binary_backend_specification>",
                f"  Formula: {self.formula_text}",
                "  Family: Bernoulli-logit",
                f"  Interface: {self.backend_interface}",
                f"  Sampling backend: {self.sampling_backend}",
                f"  Algorithm: {self.algorithm}",
                f"  Backend available: {str(self.backend_available).upper()}",
                "  Compiled: FALSE",
                "  Fit performed: FALSE",
            ]
        )


def translate_binary_model_to_brms(
    specification: BinaryModelSpecification,
) -> BinaryBackendSpecification:
    """Translate an approved binary specification to the restricted Python backend plan.

    The public name mirrors gp3bayes 0.5.0.  In Python, execution is intentionally
    adapted from the R brms/rstan stack to optional PyMC NUTS while retaining the
    same locked likelihood, link, formula, prior classes, and no-escape-hatch policy.
    """
    specification = _validate_binary_model_specification(specification)
    parameter_table = _translation_parameter_table(
        specification.priors,
        include_sigma=False,
        random_slope=specification.contract.random_slope,
    )
    prior_text = {
        str(row.parameter_class): str(row.prior)
        for row in parameter_table.itertuples(index=False)
    }
    return BinaryBackendSpecification(
        translation_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        formula=specification.formula,
        formula_text=specification.formula_text,
        family_object={"family": "Bernoulli", "link": "logit"},
        priors=prior_text,
        prior_text=prior_text,
        validated_priors=parameter_table.copy(),
        parameter_table=parameter_table,
        specification=specification,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        backend_available=_pymc_available(),
    )


@dataclass(frozen=True, slots=True)
class BinaryFit:
    """Fitted restricted hierarchical binary model and sampling provenance."""

    fit_version: str
    family: str
    model_family: str
    specification: BinaryModelSpecification
    translation: BinaryBackendSpecification
    backend_fit: Any
    backend_model: Any
    backend_interface: str
    sampling_backend: str
    algorithm: str
    sampling: Mapping[str, int | float]
    package_versions: Mapping[str, str]
    unrestricted_formula: bool = False
    fit_performed: bool = True
    diagnostics_assessed: bool = False
    posterior_adequacy_established: bool = False

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_binary_fit>",
                f"  Formula: {self.translation.formula_text}",
                "  Family: Bernoulli-logit",
                f"  Interface: {self.backend_interface}",
                f"  Sampling backend: {self.sampling_backend}",
                f"  Algorithm: {self.algorithm}",
                f"  Chains: {self.sampling['chains']}",
                f"  Iterations per chain: {self.sampling['iter']}",
                f"  Warmup per chain: {self.sampling['warmup']}",
                "  Fit performed: TRUE",
                "  Diagnostics assessed: FALSE",
                "  Posterior adequacy established: FALSE",
            ]
        )


def _run_binary_pymc(
    specification: BinaryModelSpecification,
    controls: Mapping[str, int | float],
) -> tuple[Any, Any]:
    pm = _load_pymc()

    prepared = cast(BinaryPrepared, specification.prepared)
    data = prepared.data
    contract = specification.contract
    matrix, _ = _fixed_model_matrix(data, contract)
    outcome_col = cast(str, contract.mappings["outcome"])
    participant_col = cast(str, contract.mappings["participant"])
    y = pd.to_numeric(data[outcome_col], errors="raise").to_numpy(dtype=int)

    intercept_prior = _prior_row(specification.priors, "Intercept")
    coefficient_prior = _prior_row(specification.priors, "b")
    group_sd_prior = _prior_row(specification.priors, "sd")

    participant_codes, participant_levels = pd.factorize(
        data[participant_col], sort=False
    )
    item_col = contract.mappings["item"]
    condition_col = contract.mappings["condition"]

    with pm.Model() as model:
        intercept = pm.Normal(
            "b_Intercept",
            mu=float(intercept_prior["location"]),
            sigma=float(intercept_prior["scale"]),
        )
        eta = intercept
        if matrix.shape[1] > 1:
            beta = pm.Normal(
                "b",
                mu=float(coefficient_prior["location"]),
                sigma=float(coefficient_prior["scale"]),
                shape=matrix.shape[1] - 1,
            )
            eta = eta + pm.math.dot(matrix[:, 1:], beta)

        if contract.random_slope:
            correlation_prior = _prior_row(specification.priors, "cor")
            sd_dist = pm.HalfStudentT.dist(
                nu=float(group_sd_prior["df"]),
                sigma=float(group_sd_prior["scale"]),
            )
            chol, _, _ = pm.LKJCholeskyCov(
                "participant_chol",
                n=2,
                eta=float(correlation_prior["shape"]),
                sd_dist=sd_dist,
                compute_corr=True,
            )
            z = pm.Normal(
                "participant_z",
                mu=0.0,
                sigma=1.0,
                shape=(len(participant_levels), 2),
            )
            participant_re = pm.Deterministic(
                "participant_re", pm.math.dot(z, chol.T)
            )
            if condition_col is None:
                raise GP3BayesError(
                    "`condition_col` must be supplied when `random_slope` is TRUE."
                )
            condition = pd.to_numeric(
                data[condition_col], errors="raise"
            ).to_numpy(dtype=float)
            eta = (
                eta
                + participant_re[participant_codes, 0]
                + participant_re[participant_codes, 1] * condition
            )
        else:
            participant_sd = pm.HalfStudentT(
                "sd_participant",
                nu=float(group_sd_prior["df"]),
                sigma=float(group_sd_prior["scale"]),
            )
            participant_z = pm.Normal(
                "participant_z",
                mu=0.0,
                sigma=1.0,
                shape=len(participant_levels),
            )
            eta = eta + participant_sd * participant_z[participant_codes]

        if item_col is not None:
            item_codes, item_levels = pd.factorize(data[item_col], sort=False)
            item_sd = pm.HalfStudentT(
                "sd_item",
                nu=float(group_sd_prior["df"]),
                sigma=float(group_sd_prior["scale"]),
            )
            item_z = pm.Normal(
                "item_z", mu=0.0, sigma=1.0, shape=len(item_levels)
            )
            eta = eta + item_sd * item_z[item_codes]

        pm.Bernoulli("observed", logit_p=eta, observed=y)
        idata = pm.sample(
            draws=int(controls["post_warmup_iterations"]),
            tune=int(controls["warmup"]),
            chains=int(controls["chains"]),
            cores=int(controls["cores"]),
            random_seed=int(controls["seed"]),
            progressbar=int(controls["refresh"]) > 0,
            compute_convergence_checks=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            nuts={
                "target_accept": float(controls["adapt_delta"]),
                "max_treedepth": int(controls["max_treedepth"]),
            },
        )
    return model, idata


def fit_binary_model(
    specification: BinaryModelSpecification,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> BinaryFit:
    """Fit the approved hierarchical binary model with optional PyMC NUTS."""
    specification = _validate_binary_model_specification(specification)
    controls = _validate_sampling_controls(
        chains=chains,
        iter=iter,
        warmup=warmup,
        cores=cores,
        seed=seed,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        refresh=refresh,
    )
    translation = translate_binary_model_to_brms(specification)
    _require_pymc("fit a binary model through the approved Python sampling backend")
    backend_model, backend_fit = _run_binary_pymc(
        specification, controls.as_dict()
    )
    return BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=specification,
        translation=translation,
        backend_fit=backend_fit,
        backend_model=backend_model,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling=controls.as_dict(),
        package_versions=_backend_versions(),
    )
