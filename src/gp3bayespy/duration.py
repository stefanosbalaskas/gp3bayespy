"""Hierarchical duration workflow foundation.

Python-native port of the frozen gp3bayes 0.5.0
``duration-workflow-foundation.R`` layer. The workflow is deliberately
backend-independent: simulation, deterministic preparation, prior declaration,
and prior-predictive simulation do not fit a Bayesian model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from .binary import (
    _character_vector,
    _code_condition,
    _column_basis,
    _flag,
    _integer,
    _numeric_scalar,
    _prior_row,
    _quote_name,
    _sample_sd,
    _standardize,
)
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
from .posterior import diagnose_fit as _diagnose_fit, summarise_duration as _summarise_duration
from .specification import (
    ModelSpecification,
    create_model_specification,
    create_prior_specification,
    validate_prior_specification,
)


def _validate_duration_contract(contract: ModelContract) -> ModelContract:
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")
    if contract.family != "duration":
        raise GP3BayesError("This workflow requires a duration model contract.")
    if contract.likelihood != "lognormal":
        raise GP3BayesError(
            "The duration workflow requires the approved lognormal likelihood."
        )
    if not isinstance(contract.outcome_unit, str) or not contract.outcome_unit:
        raise GP3BayesError(
            "A duration contract must record one non-empty `outcome_unit`."
        )
    return contract


def _fixed_formula_text(contract: ModelContract) -> str:
    _validate_duration_contract(contract)
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

    return np.column_stack(columns), tuple(names)


def _required_columns(contract: ModelContract) -> tuple[str, ...]:
    values: list[str] = []
    for value in (
        *contract.mappings.values(),
        *contract.predictors,
        *(contract.interaction or ()),
    ):
        if value is not None and value not in values:
            values.append(value)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class DurationSimulation:
    """Synthetic hierarchical lognormal-duration data and generating truth."""

    simulation_version: str
    family: str
    data: pd.DataFrame
    truth: Mapping[str, Any]
    random_effects: Mapping[str, pd.DataFrame | None]
    design: Mapping[str, Any]

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_duration_simulation>",
                f"  Rows: {len(self.data)}",
                f"  Participants: {self.design['n_participants']}",
                f"  Outcome unit: {self.design['outcome_unit']}",
                "  Strictly positive: TRUE",
                "  Censored: FALSE",
                "  Fit performed: FALSE",
            ]
        )


def simulate_hierarchical_duration_data(
    n_participants: int = 40,
    trials_per_participant: int = 20,
    n_items: int = 20,
    baseline_median: float = 500.0,
    condition_effect: float = math.log(1.15),
    participant_covariate_effect: float = math.log(1.08),
    trial_covariate_effect: float = math.log(1.04),
    interaction_effect: float = math.log(1.05),
    participant_sd: float = 0.35,
    item_sd: float = 0.20,
    random_slope_sd: float = 0.15,
    random_slope_cor: float = 0.0,
    residual_sd: float = 0.40,
    condition_probability: float = 0.5,
    balanced_condition: bool = True,
    include_items: bool = True,
    outcome_unit: str = "milliseconds",
    seed: int = 1,
) -> DurationSimulation:
    """Simulate governed positive uncensored lognormal duration data."""
    n_participants = _integer(n_participants, "n_participants", minimum=2)
    trials_per_participant = _integer(
        trials_per_participant, "trials_per_participant", minimum=2
    )
    n_items = _integer(n_items, "n_items", minimum=2)
    baseline_median = _numeric_scalar(
        baseline_median, "baseline_median", lower=0, lower_open=True
    )
    condition_effect = _numeric_scalar(condition_effect, "condition_effect")
    participant_covariate_effect = _numeric_scalar(
        participant_covariate_effect, "participant_covariate_effect"
    )
    trial_covariate_effect = _numeric_scalar(
        trial_covariate_effect, "trial_covariate_effect"
    )
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
    residual_sd = _numeric_scalar(
        residual_sd, "residual_sd", lower=0, lower_open=True
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
    if not isinstance(outcome_unit, str) or not outcome_unit:
        raise GP3BayesError("`outcome_unit` must be one non-empty character value.")
    seed = _integer(seed, "seed", minimum=0)

    rng = np.random.RandomState(seed)
    participant_levels = np.array(
        [f"p{index:03d}" for index in range(1, n_participants + 1)], dtype=object
    )
    participant_id = np.repeat(participant_levels, trials_per_participant)
    trial_id = np.tile(np.arange(1, trials_per_participant + 1), n_participants)
    n_rows = len(participant_id)

    if balanced_condition:
        base = np.resize(np.array([-0.5, 0.5], dtype=float), trials_per_participant)
        condition_code = np.concatenate(
            [rng.permutation(base) for _ in range(n_participants)]
        )
    else:
        condition_code = np.where(
            rng.uniform(size=n_rows) < condition_probability, 0.5, -0.5
        )

    participant_covariate_by_id = _standardize(rng.normal(size=n_participants))
    participant_index = pd.Categorical(
        participant_id, categories=participant_levels
    ).codes
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
        item_levels = np.array(
            [f"i{index:03d}" for index in range(1, n_items + 1)], dtype=object
        )
        participant_offsets = np.repeat(
            np.arange(n_participants), trials_per_participant
        )
        item_index = (trial_id + participant_offsets - 1) % n_items
        item_id = item_levels[item_index]
        item_effect_by_id = rng.normal(loc=0.0, scale=item_sd, size=n_items)
        item_effect = item_effect_by_id[item_index]

    linear_predictor = (
        math.log(baseline_median)
        + condition_effect * condition_code
        + participant_covariate_effect * participant_covariate
        + trial_covariate_effect * trial_covariate
        + interaction_effect * condition_code * participant_covariate
        + participant_intercept[participant_index]
        + participant_slope[participant_index] * condition_code
        + item_effect
    )
    duration = rng.lognormal(
        mean=linear_predictor, sigma=residual_sd, size=n_rows
    )

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
        "duration": duration,
        "true_median": np.exp(linear_predictor),
        "true_mean": np.exp(linear_predictor + residual_sd**2 / 2),
    }
    if include_items:
        data_dict["item_id"] = item_id
    preferred = [
        "participant_id",
        *(["item_id"] if include_items else []),
        "trial_id",
        "condition",
        "participant_covariate",
        "trial_covariate",
        "duration",
        "true_median",
        "true_mean",
    ]
    data = cast(pd.DataFrame, pd.DataFrame(data_dict).loc[:, preferred])

    truth: dict[str, Any] = {
        "fixed_effects": {
            "(Intercept)": math.log(baseline_median),
            "condition": condition_effect,
            "participant_covariate": participant_covariate_effect,
            "trial_covariate": trial_covariate_effect,
            "condition:participant_covariate": interaction_effect,
        },
        "baseline_median": baseline_median,
        "participant_sd": participant_sd,
        "item_sd": item_sd if include_items else 0.0,
        "random_slope_sd": random_slope_sd,
        "random_slope_cor": random_slope_cor,
        "residual_sd": residual_sd,
        "condition_coding": {"control": -0.5, "treatment": 0.5},
        "condition_probability": condition_probability,
        "balanced_condition": balanced_condition,
        "outcome_unit": outcome_unit,
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
        "outcome_unit": outcome_unit,
        "censored": False,
    }
    return DurationSimulation(
        simulation_version="0.1",
        family="duration",
        data=data,
        truth=truth,
        random_effects={"participant": participant_re, "item": item_re},
        design=design,
    )


@dataclass(frozen=True, slots=True)
class DurationPrepared:
    """Deterministically prepared strictly-positive duration analysis data."""

    preparation_version: str
    family: str
    data: pd.DataFrame
    source_contract: ModelContract
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
    outcome_unit: str
    contains_data: bool = True
    backend: str = "none"
    fit_performed: bool = False

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_duration_prepared>",
                f"  Analysis rows: {self.n_analysis_rows}",
                f"  Rows removed: {self.rows_removed}",
                f"  Outcome unit: {self.outcome_unit}",
                f"  Readiness status: {self.audit.status}",
                "  Fit performed: FALSE",
            ]
        )


def prepare_hierarchical_duration_data(
    data: pd.DataFrame,
    contract: ModelContract,
    condition_levels: Sequence[object] | None = None,
    condition_coding: Sequence[float] = (-0.5, 0.5),
    scale_predictors: Sequence[str] | str = (),
    scale_time: bool = False,
    outcome_multiplier: float = 1.0,
    converted_unit: str | None = None,
    missing: str = "error",
) -> DurationPrepared:
    """Validate, explicitly convert, scale, and readiness-gate duration data."""
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    _validate_duration_contract(contract)
    if missing not in {"error", "drop"}:
        raise GP3BayesError('`missing` must be either "error" or "drop".')
    scale_predictors_tuple = _character_vector(scale_predictors, "scale_predictors")
    scale_time = _flag(scale_time, "scale_time")
    outcome_multiplier = _numeric_scalar(
        outcome_multiplier, "outcome_multiplier", lower=0, lower_open=True
    )
    if any(value not in contract.predictors for value in scale_predictors_tuple):
        raise GP3BayesError(
            "Every scaled predictor must be declared in the model contract."
        )

    if not math.isclose(outcome_multiplier, 1.0):
        if not isinstance(converted_unit, str) or not converted_unit:
            raise GP3BayesError(
                "`converted_unit` must be supplied when `outcome_multiplier` is not one."
            )
    elif converted_unit is not None and (
        not isinstance(converted_unit, str) or not converted_unit
    ):
        raise GP3BayesError(
            "`converted_unit` must be NULL or one non-empty character value."
        )

    required = _required_columns(contract)
    absent = [value for value in required if value not in data.columns]
    if absent:
        raise GP3BayesError(
            "Required duration columns were not found: " + ", ".join(absent) + "."
        )

    required_frame = cast(pd.DataFrame, data.loc[:, list(required)])
    complete_rows = ~required_frame.isna().any(axis=1)
    dropped_positions = [
        index + 1 for index, complete in enumerate(complete_rows) if not complete
    ]
    if dropped_positions and missing == "error":
        raise GP3BayesError(
            "Missing values were found in required duration columns. "
            'Use `missing = "drop"` only after an explicit exclusion decision.'
        )
    working = (
        cast(pd.DataFrame, data.loc[complete_rows]).copy()
        if dropped_positions
        else data.copy()
    )

    outcome_column = cast(str, contract.mappings["outcome"])
    raw_outcome = pd.to_numeric(working[outcome_column], errors="coerce")
    values = raw_outcome.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise GP3BayesError(
            "The duration outcome must contain only finite numeric values."
        )
    values = values * outcome_multiplier
    if np.any(values <= 0):
        raise GP3BayesError(
            "The duration outcome must be strictly positive. "
            "Zero and negative values are unsupported."
        )
    working[outcome_column] = values

    analysis_contract = (
        replace(contract, outcome_unit=converted_unit)
        if converted_unit is not None
        else contract
    )
    analysis_unit = cast(str, analysis_contract.outcome_unit)
    transformations: dict[str, Any] = {
        "outcome": {
            "source_unit": contract.outcome_unit,
            "analysis_unit": analysis_unit,
            "multiplier": outcome_multiplier,
            "strictly_positive": True,
            "finite": True,
            "censored": False,
        },
        "condition": None,
        "scaled_columns": {},
        "missing": {
            "action": missing,
            "dropped_row_positions": tuple(dropped_positions),
        },
    }

    condition_column = analysis_contract.mappings["condition"]
    if condition_column is not None:
        coded, source_levels, coding = _code_condition(
            working[condition_column], condition_levels, condition_coding
        )
        working[condition_column] = coded
        transformations["condition"] = {
            "source_levels": source_levels,
            "coding": coding,
        }

    scaling_columns = list(scale_predictors_tuple)
    time_column = analysis_contract.mappings["time"]
    if scale_time and time_column is not None and time_column not in scaling_columns:
        scaling_columns.append(time_column)
    scaled_registry = cast(dict[str, dict[str, float | str]], transformations["scaled_columns"])
    for column in scaling_columns:
        series = working[column]
        if not pd.api.types.is_numeric_dtype(series.dtype):
            raise GP3BayesError(
                f"Only numeric predictors can be scaled. `{column}` is not numeric."
            )
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        center = float(np.mean(numeric))
        scale_value = _sample_sd(numeric)
        if not math.isfinite(scale_value) or scale_value <= 0:
            raise GP3BayesError(
                f"The declared scaling column `{column}` has zero or undefined "
                "standard deviation."
            )
        working[column] = (numeric - center) / scale_value
        scaled_registry[column] = {
            "action": "centre_and_scale",
            "centre": center,
            "scale": scale_value,
        }

    audit = audit_model_readiness(working, analysis_contract)
    if not audit.ready:
        raise GP3BayesError(
            "The prepared duration data did not pass the readiness gate."
        )

    fixed_formula = _fixed_formula_text(analysis_contract)
    model_matrix, matrix_columns = _fixed_model_matrix(working, analysis_contract)
    if np.linalg.matrix_rank(model_matrix) < model_matrix.shape[1]:
        raise GP3BayesError("The prepared fixed-effects design matrix is rank deficient.")

    condition_transform = transformations["condition"]
    if condition_transform is None:
        condition_value = "not applicable"
    else:
        condition_map = cast(Mapping[str, float], condition_transform["coding"])
        condition_value = ", ".join(
            f"{key}={value:g}" for key, value in condition_map.items()
        )
    decision_log = pd.DataFrame(
        {
            "decision": [
                "outcome_validation",
                "unit_conversion",
                "missing_values",
                "condition_coding",
                "predictor_scaling",
            ],
            "value": [
                "strictly positive finite uncensored",
                f"{contract.outcome_unit} x {outcome_multiplier:g} -> {analysis_unit}",
                f"{missing}; rows removed = {len(dropped_positions)}",
                condition_value,
                ", ".join(scaling_columns) if scaling_columns else "none",
            ],
        }
    )

    return DurationPrepared(
        preparation_version="0.1",
        family="duration",
        data=working,
        source_contract=contract,
        contract=analysis_contract,
        audit=audit,
        transformations=transformations,
        decision_log=decision_log,
        fixed_formula=fixed_formula,
        fixed_formula_text=fixed_formula,
        model_matrix_columns=matrix_columns,
        n_input_rows=len(data),
        n_analysis_rows=len(working),
        rows_removed=len(dropped_positions),
        outcome_unit=analysis_unit,
    )


@dataclass(frozen=True, slots=True)
class DurationModelSpecification(ModelSpecification):
    """Backend-independent duration model specification."""

    duration_workflow_version: str = "0.1"
    prepared: DurationPrepared | None = None
    fixed_formula: str = ""
    fixed_formula_text: str = ""
    model_matrix_columns: tuple[str, ...] = ()
    outcome_unit: str = ""
    contains_data: bool = True
    fitting_engine: str = "none"
    backend_dependency: str = "none"
    unrestricted_formula: bool = False
    prior_predictive_performed: bool = False

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_duration_model_specification>",
                f"  Formula: {self.formula_text}",
                f"  Family: {self.contract.likelihood}",
                f"  Outcome unit: {self.outcome_unit}",
                f"  Readiness: {self.readiness_status}",
                "  Fit performed: FALSE",
            ]
        )


def specify_duration_model(
    prepared: DurationPrepared,
    baseline: float,
    intercept_scale: float = 1.0,
    coefficient_scale: float = 0.5,
    group_sd_scale: float = 1.0,
    residual_scale: float = 1.0,
    correlation_eta: float = 2.0,
    student_df: float = 3.0,
) -> DurationModelSpecification:
    """Combine prepared duration data with approved lognormal priors."""
    if not isinstance(prepared, DurationPrepared):
        raise GP3BayesError("`prepared` must inherit from `gp3bayes_duration_prepared`.")
    _validate_duration_contract(prepared.contract)
    if not prepared.audit.ready:
        raise GP3BayesError("`prepared$audit` is not ready for specification.")

    priors = create_prior_specification(
        prepared.contract,
        baseline=baseline,
        intercept_scale=intercept_scale,
        coefficient_scale=coefficient_scale,
        group_sd_scale=group_sd_scale,
        residual_scale=residual_scale,
        correlation_eta=correlation_eta,
        student_df=student_df,
    )
    core = create_model_specification(prepared.contract, prepared.audit, priors)
    core_values = {
        field.name: getattr(core, field.name) for field in fields(ModelSpecification)
    }
    return DurationModelSpecification(
        **core_values,
        duration_workflow_version="0.1",
        prepared=prepared,
        fixed_formula=prepared.fixed_formula,
        fixed_formula_text=prepared.fixed_formula_text,
        model_matrix_columns=prepared.model_matrix_columns,
        outcome_unit=prepared.outcome_unit,
    )


def _duration_summary(
    y: np.ndarray,
    condition: np.ndarray | None,
    participant: np.ndarray,
    item: np.ndarray | None,
) -> dict[str, float]:
    valid = np.isfinite(y) & (y > 0)
    nonfinite_fraction = float(np.mean(~valid))
    if not valid.any():
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

    y_valid = y[valid]
    participant_valid = participant[valid]
    log_y = np.log(y_valid)
    participant_levels = pd.unique(participant_valid)
    participant_medians = np.array(
        [
            float(np.median(log_y[participant_valid == level]))
            for level in participant_levels
        ]
    )

    condition_ratio = math.nan
    if condition is not None:
        condition_valid = condition[valid]
        levels = np.sort(np.unique(condition_valid))
        if len(levels) == 2:
            medians = [
                float(np.median(y_valid[condition_valid == level])) for level in levels
            ]
            if all(math.isfinite(value) for value in medians) and medians[0] > 0:
                condition_ratio = medians[1] / medians[0]

    item_log_median_sd = math.nan
    if item is not None:
        item_valid = item[valid]
        item_levels = pd.unique(item_valid)
        item_medians = np.array(
            [float(np.median(log_y[item_valid == level])) for level in item_levels]
        )
        if len(item_medians) > 1:
            item_log_median_sd = _sample_sd(item_medians)

    mean_y = float(np.mean(y_valid))
    return {
        "median": float(np.median(y_valid)),
        "mean": mean_y,
        "q90": float(np.quantile(y_valid, 0.90, method="median_unbiased")),
        "q99": float(np.quantile(y_valid, 0.99, method="median_unbiased")),
        "coefficient_of_variation": (
            _sample_sd(y_valid) / mean_y
            if len(y_valid) > 1 and math.isfinite(mean_y) and mean_y > 0
            else math.nan
        ),
        "condition_median_ratio": condition_ratio,
        "participant_log_median_sd": (
            _sample_sd(participant_medians)
            if len(participant_medians) > 1
            else math.nan
        ),
        "item_log_median_sd": item_log_median_sd,
        "nonfinite_fraction": nonfinite_fraction,
    }


@dataclass(frozen=True, slots=True)
class DurationPriorPredictiveCheck:
    """Backend-independent prior-predictive audit for duration specifications."""

    check_version: str
    family: str
    draws: int
    seed: int
    outcome_unit: str
    summaries: pd.DataFrame
    checks: pd.DataFrame
    thresholds: Mapping[str, Any]
    adequate: bool
    backend: str = "none"
    fitting_performed: bool = False
    posterior_adequacy_established: bool = False

    def __repr__(self) -> str:
        failures = int((self.checks["status"] == "fail").sum())
        return "\n".join(
            [
                "<gp3bayes_duration_prior_predictive_check>",
                f"  Draws: {self.draws}",
                f"  Outcome unit: {self.outcome_unit}",
                f"  Adequate: {str(self.adequate).upper()}",
                f"  Failed checks: {failures}",
                "  Backend: none",
                "  Fit performed: FALSE",
                "  Posterior adequacy established: FALSE",
            ]
        )


def _positive_pair(
    value: Sequence[float] | None, baseline: float
) -> tuple[float, float]:
    values = (baseline * 0.1, baseline * 10.0) if value is None else tuple(value)
    pair = tuple(float(item) for item in values)
    if (
        len(pair) != 2
        or not all(math.isfinite(item) and item > 0 for item in pair)
        or pair[0] >= pair[1]
    ):
        raise GP3BayesError(
            "`plausible_median` must be an increasing pair of positive finite values."
        )
    return pair[0], pair[1]


def check_duration_prior_predictive(
    specification: DurationModelSpecification,
    draws: int = 500,
    seed: int = 1,
    plausible_median: Sequence[float] | None = None,
    maximum_q99: float | None = None,
    maximum_cv: float = 5.0,
    maximum_condition_ratio: float = 10.0,
    maximum_extreme_probability: float = 0.25,
) -> DurationPriorPredictiveCheck:
    """Simulate lognormal prior predictive outcomes without fitting a model."""
    if not isinstance(specification, DurationModelSpecification):
        raise GP3BayesError(
            "`specification` must inherit from "
            "`gp3bayes_duration_model_specification`."
        )
    prepared = specification.prepared
    if not isinstance(prepared, DurationPrepared):
        raise GP3BayesError(
            "`specification$prepared` must inherit from `gp3bayes_duration_prepared`."
        )
    validate_prior_specification(specification.priors, specification.contract)
    draws = _integer(draws, "draws", minimum=50)
    seed = _integer(seed, "seed", minimum=0)
    baseline = float(specification.priors.baseline)
    plausible = _positive_pair(plausible_median, baseline)
    q99_limit = _numeric_scalar(
        baseline * 50 if maximum_q99 is None else maximum_q99,
        "maximum_q99",
        lower=0,
        lower_open=True,
    )
    maximum_cv = _numeric_scalar(
        maximum_cv, "maximum_cv", lower=0, lower_open=True
    )
    maximum_condition_ratio = _numeric_scalar(
        maximum_condition_ratio,
        "maximum_condition_ratio",
        lower=1,
        lower_open=True,
    )
    maximum_extreme_probability = _numeric_scalar(
        maximum_extreme_probability,
        "maximum_extreme_probability",
        lower=0,
        upper=1,
    )

    data = prepared.data
    contract = prepared.contract
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
        condition = pd.to_numeric(
            data[condition_column], errors="raise"
        ).to_numpy(dtype=float)

    intercept_prior = _prior_row(specification.priors, "Intercept")
    coefficient_prior = _prior_row(specification.priors, "b")
    group_prior = _prior_row(specification.priors, "sd")
    sigma_prior = _prior_row(specification.priors, "sigma")
    correlation_prior = (
        _prior_row(specification.priors, "cor") if contract.random_slope else None
    )

    rng = np.random.RandomState(seed)
    rows: list[dict[str, float]] = []
    for _ in range(draws):
        intercept = rng.normal(
            loc=float(intercept_prior["location"]),
            scale=float(intercept_prior["scale"]),
        )
        coefficient_count = model_matrix.shape[1] - 1
        linear_predictor = np.full(len(data), intercept, dtype=float)
        if coefficient_count > 0:
            coefficients = rng.normal(
                loc=float(coefficient_prior["location"]),
                scale=float(coefficient_prior["scale"]),
                size=coefficient_count,
            )
            linear_predictor = (
                linear_predictor + model_matrix[:, 1:] @ coefficients
            )

        group_scale = float(group_prior["scale"])
        group_df = float(group_prior["df"])
        participant_intercept_sd = abs(rng.standard_t(group_df)) * group_scale
        participant_slope_sd = (
            abs(rng.standard_t(group_df)) * group_scale
            if contract.random_slope
            else 0.0
        )
        z_intercept = rng.normal(size=participant_count)
        z_slope = rng.normal(size=participant_count)
        correlation = 0.0
        if contract.random_slope:
            assert correlation_prior is not None
            shape = float(correlation_prior["shape"])
            correlation = 2 * rng.beta(shape, shape) - 1
        participant_intercept = participant_intercept_sd * z_intercept
        participant_slope = participant_slope_sd * (
            correlation * z_intercept
            + math.sqrt(1 - correlation**2) * z_slope
        )
        linear_predictor = (
            linear_predictor + participant_intercept[participant_index]
        )
        if contract.random_slope:
            assert condition is not None
            linear_predictor = (
                linear_predictor
                + participant_slope[participant_index] * condition
            )

        if item is not None:
            assert item_index is not None
            item_effect_sd = abs(rng.standard_t(group_df)) * group_scale
            item_effect = rng.normal(loc=0.0, scale=item_effect_sd, size=item_count)
            linear_predictor = linear_predictor + item_effect[item_index]

        sigma = abs(rng.standard_t(float(sigma_prior["df"]))) * float(
            sigma_prior["scale"]
        )
        y = rng.lognormal(mean=linear_predictor, sigma=sigma, size=len(data))
        rows.append(_duration_summary(y, condition, participant, item))

    summaries = pd.DataFrame(rows)
    median_violation = float(
        np.mean(
            (summaries["median"] < plausible[0])
            | (summaries["median"] > plausible[1])
        )
    )
    q99_violation = float(np.mean(summaries["q99"] > q99_limit))
    cv_values = summaries["coefficient_of_variation"].to_numpy(dtype=float)
    cv_violation = (
        1.0
        if np.isnan(cv_values).all()
        else float(np.mean(cv_values[~np.isnan(cv_values)] > maximum_cv))
    )
    ratios = summaries["condition_median_ratio"].to_numpy(dtype=float)
    if np.isnan(ratios).all():
        condition_violation = math.nan
    else:
        finite = ratios[~np.isnan(ratios)]
        condition_violation = float(
            np.mean(
                (finite > maximum_condition_ratio)
                | (finite < 1 / maximum_condition_ratio)
            )
        )
    nonfinite_violation = float(
        np.mean(summaries["nonfinite_fraction"] > 0)
    )

    probabilities = [
        median_violation,
        q99_violation,
        cv_violation,
        condition_violation,
        nonfinite_violation,
    ]
    statuses = [
        "not_applicable"
        if math.isnan(probability)
        else (
            "pass"
            if probability <= maximum_extreme_probability
            else "fail"
        )
        for probability in probabilities
    ]
    checks = pd.DataFrame(
        {
            "check": [
                "overall_median",
                "upper_tail_q99",
                "coefficient_of_variation",
                "condition_median_ratio",
                "nonfinite_predictions",
            ],
            "violation_probability": probabilities,
            "maximum_probability": [maximum_extreme_probability] * 5,
            "status": statuses,
        }
    )
    adequate = bool(checks["status"].isin(["pass", "not_applicable"]).all())
    return DurationPriorPredictiveCheck(
        check_version="0.1",
        family="duration",
        draws=draws,
        seed=seed,
        outcome_unit=prepared.outcome_unit,
        summaries=summaries,
        checks=checks,
        thresholds={
            "plausible_median": plausible,
            "maximum_q99": q99_limit,
            "maximum_cv": maximum_cv,
            "maximum_condition_ratio": maximum_condition_ratio,
            "maximum_extreme_probability": maximum_extreme_probability,
        },
        adequate=adequate,
    )


def _validate_duration_model_specification(
    specification: DurationModelSpecification,
) -> DurationModelSpecification:
    if not isinstance(specification, DurationModelSpecification):
        raise GP3BayesError(
            "`specification` must inherit from `gp3bayes_duration_model_specification`."
        )
    _validate_duration_contract(specification.contract)
    if not isinstance(specification.prepared, DurationPrepared):
        raise GP3BayesError(
            "`specification$prepared` must inherit from `gp3bayes_duration_prepared`."
        )
    if not specification.audit.ready:
        raise GP3BayesError("`specification$audit` must pass the readiness gate.")
    return specification


@dataclass(frozen=True, slots=True)
class DurationBackendSpecification:
    """Restricted Python backend translation of the R duration brms contract."""

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
    specification: DurationModelSpecification
    outcome_unit: str
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
        "the restricted lognormal formula/prior contract and executes with optional PyMC NUTS."
    )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_duration_backend_specification>",
                f"  Formula: {self.formula_text}",
                "  Family: lognormal",
                f"  Outcome unit: {self.outcome_unit}",
                f"  Interface: {self.backend_interface}",
                f"  Sampling backend: {self.sampling_backend}",
                f"  Algorithm: {self.algorithm}",
                f"  Backend available: {str(self.backend_available).upper()}",
                "  Compiled: FALSE",
                "  Fit performed: FALSE",
            ]
        )


def translate_duration_model_to_brms(
    specification: DurationModelSpecification,
) -> DurationBackendSpecification:
    """Translate an approved duration specification to the Python backend plan."""
    specification = _validate_duration_model_specification(specification)
    parameter_table = _translation_parameter_table(
        specification.priors,
        include_sigma=True,
        random_slope=specification.contract.random_slope,
    )
    prior_text = {
        str(row.parameter_class): str(row.prior)
        for row in parameter_table.itertuples(index=False)
    }
    return DurationBackendSpecification(
        translation_version="0.1",
        family="duration",
        model_family="hierarchical_lognormal_duration",
        formula=specification.formula,
        formula_text=specification.formula_text,
        family_object={"family": "lognormal", "link": "identity"},
        priors=prior_text,
        prior_text=prior_text,
        validated_priors=parameter_table.copy(),
        parameter_table=parameter_table,
        specification=specification,
        outcome_unit=specification.outcome_unit,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        backend_available=_pymc_available(),
    )


@dataclass(frozen=True, slots=True)
class DurationFit:
    """Fitted restricted hierarchical lognormal duration model."""

    fit_version: str
    family: str
    model_family: str
    specification: DurationModelSpecification
    translation: DurationBackendSpecification
    backend_fit: Any
    backend_model: Any
    outcome_unit: str
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
                "<gp3bayes_duration_fit>",
                f"  Formula: {self.translation.formula_text}",
                "  Family: lognormal",
                f"  Outcome unit: {self.outcome_unit}",
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


def _run_duration_pymc(
    specification: DurationModelSpecification,
    controls: Mapping[str, int | float],
) -> tuple[Any, Any]:
    pm = _load_pymc()

    prepared = cast(DurationPrepared, specification.prepared)
    data = prepared.data
    contract = specification.contract
    matrix, _ = _fixed_model_matrix(data, contract)
    outcome_col = cast(str, contract.mappings["outcome"])
    participant_col = cast(str, contract.mappings["participant"])
    y = pd.to_numeric(data[outcome_col], errors="raise").to_numpy(dtype=float)

    intercept_prior = _prior_row(specification.priors, "Intercept")
    coefficient_prior = _prior_row(specification.priors, "b")
    group_sd_prior = _prior_row(specification.priors, "sd")
    sigma_prior = _prior_row(specification.priors, "sigma")

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

        sigma = pm.HalfStudentT(
            "sigma",
            nu=float(sigma_prior["df"]),
            sigma=float(sigma_prior["scale"]),
        )
        pm.LogNormal("observed", mu=eta, sigma=sigma, observed=y)
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


def fit_duration_model(
    specification: DurationModelSpecification,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> DurationFit:
    """Fit the approved lognormal duration model with optional PyMC NUTS."""
    specification = _validate_duration_model_specification(specification)
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
    translation = translate_duration_model_to_brms(specification)
    _require_pymc("fit a duration model through the approved Python sampling backend")
    backend_model, backend_fit = _run_duration_pymc(
        specification, controls.as_dict()
    )
    return DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_lognormal_duration",
        specification=specification,
        translation=translation,
        backend_fit=backend_fit,
        backend_model=backend_model,
        outcome_unit=specification.outcome_unit,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling=controls.as_dict(),
        package_versions=_backend_versions(),
    )


def diagnose_duration_fit(
    fit: DurationFit,
    rhat_pass: float = 1.01,
    rhat_fail: float = 1.05,
    ess_per_chain_pass: float = 100,
    ess_per_chain_fail: float = 50,
    maximum_treedepth_fraction: float = 0.01,
    ebfmi_pass: float = 0.30,
    ebfmi_fail: float = 0.20,
):
    """Apply the frozen sampling-diagnostic thresholds to a duration fit."""
    if not isinstance(fit, DurationFit):
        raise GP3BayesError("`fit` must inherit from `gp3bayes_fit`.")
    return _diagnose_fit(
        fit,
        family="duration",
        rhat_pass=rhat_pass,
        rhat_fail=rhat_fail,
        ess_per_chain_pass=ess_per_chain_pass,
        ess_per_chain_fail=ess_per_chain_fail,
        maximum_treedepth_fraction=maximum_treedepth_fraction,
        ebfmi_pass=ebfmi_pass,
        ebfmi_fail=ebfmi_fail,
    )


def summarise_duration_posterior(
    fit: DurationFit,
    probability: float = 0.95,
    variables: Sequence[str] | str | None = None,
):
    """Summarise duration posterior parameters and median-ratio transforms."""
    if not isinstance(fit, DurationFit):
        raise GP3BayesError("`fit` must inherit from `gp3bayes_fit`.")
    return _summarise_duration(fit, probability=probability, variables=variables)
