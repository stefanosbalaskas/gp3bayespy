"""Backend-independent model-readiness audits.

This module ports the observable-data readiness semantics of gp3bayes 0.5.0.
Readiness is a gate on data structure only; it does not establish model
adequacy, convergence, predictive validity, causal identification, or
substantive validity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .contracts import ModelContract
from .exceptions import GP3BayesError


@dataclass(frozen=True, slots=True)
class AuditCheck:
    """One readiness check emitted by :func:`audit_model_readiness`."""

    check_id: str
    category: str
    status: str
    message: str
    n_affected: int | None = None


@dataclass(frozen=True, slots=True)
class ReadinessAudit:
    """Structured result of a backend-independent readiness audit."""

    audit_version: str
    family: str
    model_family: str
    ready: bool
    status: str
    n_rows: int
    n_columns: int
    status_counts: dict[str, int]
    checks: pd.DataFrame
    columns: dict[str, Any]
    observed: dict[str, int | None]
    contract: ModelContract

    def __repr__(self) -> str:
        counts = self.status_counts
        lines = [
            "<gp3bayes_readiness_audit>",
            f"  Family: {self.family}",
            f"  Rows: {self.n_rows}",
            f"  Status: {self.status}",
            f"  Ready: {str(self.ready).upper()}",
            (
                "  Checks: "
                f"{counts['pass']} passed, "
                f"{counts['warn']} warnings, "
                f"{counts['fail']} failures"
            ),
        ]
        issues = self.checks[self.checks["status"] != "pass"]
        if not issues.empty:
            lines.append("  Issues:")
            for row in issues.itertuples(index=False):
                status = cast(str, row.status)
                check_id = cast(str, row.check_id)
                message = cast(str, row.message)
                lines.append(f"    [{status.upper()}] {check_id}: {message}")
        return "\n".join(lines)


AddCheck = Callable[[str, str, str, str, int | None], None]


def _supported_scalar(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return not isinstance(value, complex)
    return False


def _supported_identifier(series: pd.Series) -> bool:
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return False
    if ptypes.is_timedelta64_dtype(series.dtype):
        return False
    if ptypes.is_complex_dtype(series.dtype):
        return False
    if (
        ptypes.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or ptypes.is_bool_dtype(series.dtype)
        or ptypes.is_numeric_dtype(series.dtype)
    ):
        if ptypes.is_object_dtype(series.dtype):
            return all(_supported_scalar(value) for value in series.tolist())
        return True
    if ptypes.is_object_dtype(series.dtype):
        return all(_supported_scalar(value) for value in series.tolist())
    return False


def _supported_predictor(series: pd.Series) -> bool:
    return _supported_identifier(series)


def _is_categorical(series: pd.Series) -> bool:
    return (
        ptypes.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or ptypes.is_bool_dtype(series.dtype)
        or ptypes.is_object_dtype(series.dtype)
    ) and not ptypes.is_numeric_dtype(series.dtype)


def _n_unique(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def _observed_levels(data: pd.DataFrame, column: str | None) -> int | None:
    if column is None or column not in data.columns:
        return None
    series = data[column]
    if not isinstance(series, pd.Series):
        return None
    return _n_unique(series)


def _validate_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    if any(
        column is None or not isinstance(column, str) or not column
        for column in data.columns
    ):
        raise GP3BayesError("`data` must have non-empty column names.")
    if data.columns.duplicated().any():
        raise GP3BayesError("`data` must not contain duplicated column names.")


def _validate_contract(contract: ModelContract) -> None:
    if not isinstance(contract, ModelContract):
        raise GP3BayesError(
            "`contract` must inherit from `gp3bayes_model_contract`."
        )
    required_mappings = {
        "outcome",
        "participant",
        "item",
        "trial",
        "condition",
        "time",
    }
    missing = required_mappings.difference(contract.mappings)
    if missing:
        names = ", ".join(sorted(missing))
        raise GP3BayesError(f"`contract$mappings` is missing: {names}.")


def _audit_identifier_types(
    data: pd.DataFrame,
    mappings: Mapping[str, str | None],
    add: AddCheck,
) -> None:
    columns = [
        column
        for key in ("participant", "item", "trial")
        if (column := mappings[key]) is not None and column in data.columns
    ]
    if not columns:
        add(
            "identifier_types",
            "identifiers",
            "fail",
            "No declared identifier columns are available.",
            0,
        )
        return
    unsupported = [
        column for column in columns if not _supported_identifier(data[column])
    ]
    if unsupported:
        add(
            "identifier_types",
            "identifiers",
            "fail",
            "Unsupported identifier column types: " + ", ".join(unsupported) + ".",
            len(unsupported),
        )
    else:
        add(
            "identifier_types",
            "identifiers",
            "pass",
            "Declared identifier columns use supported vector types.",
            None,
        )


def _audit_binary_outcome(series: pd.Series, add: AddCheck) -> None:
    supported = ptypes.is_bool_dtype(series.dtype) or (
        ptypes.is_numeric_dtype(series.dtype)
        and not ptypes.is_complex_dtype(series.dtype)
    )
    if not supported:
        add(
            "outcome_type",
            "outcome",
            "fail",
            "The binary outcome must be a logical or numeric zero-one vector.",
            len(series),
        )
        return

    add(
        "outcome_type",
        "outcome",
        "pass",
        "The binary outcome uses a supported vector type.",
        None,
    )
    numeric = pd.to_numeric(series, errors="coerce")
    observed = series.notna()
    invalid = observed & (~np.isfinite(numeric) | ~numeric.isin([0, 1]))
    n_invalid = int(invalid.sum())
    if n_invalid:
        add(
            "outcome_values",
            "outcome",
            "fail",
            f"{n_invalid} binary outcome values are not finite zero-one values.",
            n_invalid,
        )
    else:
        add(
            "outcome_values",
            "outcome",
            "pass",
            "All observed binary outcome values are zero or one.",
            None,
        )

    supported_values = sorted(
        set(float(value) for value in numeric[observed & ~invalid].tolist())
    )
    if supported_values == [0.0, 1.0]:
        add(
            "outcome_support",
            "outcome",
            "pass",
            "Both binary outcome classes are observed.",
            None,
        )
    else:
        add(
            "outcome_support",
            "outcome",
            "fail",
            "Both zero and one must be observed in the binary outcome.",
            len(supported_values),
        )


def _audit_duration_outcome(series: pd.Series, add: AddCheck) -> None:
    supported = (
        ptypes.is_numeric_dtype(series.dtype)
        and not ptypes.is_complex_dtype(series.dtype)
        and not ptypes.is_bool_dtype(series.dtype)
    )
    if not supported:
        add(
            "outcome_type",
            "outcome",
            "fail",
            "The duration outcome must be a numeric vector.",
            len(series),
        )
        return

    add(
        "outcome_type",
        "outcome",
        "pass",
        "The duration outcome uses a supported numeric type.",
        None,
    )
    observed = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    non_finite = observed & ~np.isfinite(numeric)
    n_non_finite = int(non_finite.sum())
    if n_non_finite:
        add(
            "duration_finite",
            "outcome",
            "fail",
            f"{n_non_finite} observed durations are not finite.",
            n_non_finite,
        )
    else:
        add(
            "duration_finite",
            "outcome",
            "pass",
            "All observed durations are finite.",
            None,
        )

    non_positive = observed & np.isfinite(numeric) & (numeric <= 0)
    n_non_positive = int(non_positive.sum())
    if n_non_positive:
        add(
            "duration_positive",
            "outcome",
            "fail",
            f"{n_non_positive} observed durations are not strictly positive.",
            n_non_positive,
        )
    else:
        add(
            "duration_positive",
            "outcome",
            "pass",
            "All observed durations are strictly positive.",
            None,
        )

    valid = numeric[observed & np.isfinite(numeric) & (numeric > 0)]
    n_valid_unique = int(valid.nunique())
    if n_valid_unique >= 2:
        add(
            "outcome_support",
            "outcome",
            "pass",
            "The duration outcome contains observable variation.",
            None,
        )
    else:
        add(
            "outcome_support",
            "outcome",
            "fail",
            "The duration outcome must contain at least two values.",
            n_valid_unique,
        )


def _audit_participant_structure(
    data: pd.DataFrame, participant_col: str, add: AddCheck
) -> None:
    if participant_col not in data.columns:
        return
    participant = data[participant_col]
    if not _supported_identifier(participant):
        return

    n_participants = _n_unique(participant)
    if n_participants >= 2:
        add(
            "participant_levels",
            "repeated_measures",
            "pass",
            f"{n_participants} participants are observed.",
            None,
        )
    else:
        add(
            "participant_levels",
            "repeated_measures",
            "fail",
            "At least two participants must be observed.",
            n_participants,
        )

    counts = participant.value_counts(dropna=True)
    if counts.empty or bool((counts <= 1).all()):
        add(
            "repeated_measurement",
            "repeated_measures",
            "fail",
            "No participant contributes repeated observations.",
            len(counts),
        )
    elif bool((counts <= 1).any()):
        n_singletons = int((counts <= 1).sum())
        add(
            "repeated_measurement",
            "repeated_measures",
            "warn",
            f"{n_singletons} participants contribute only one observation.",
            n_singletons,
        )
    else:
        add(
            "repeated_measurement",
            "repeated_measures",
            "pass",
            "Every participant contributes repeated observations.",
            None,
        )


def _audit_item_structure(
    data: pd.DataFrame,
    participant_col: str,
    item_col: str | None,
    add: AddCheck,
) -> None:
    if item_col is None:
        add(
            "item_structure",
            "items",
            "pass",
            "No item identifier was declared.",
            None,
        )
        return
    if item_col not in data.columns:
        return
    item = data[item_col]
    if not _supported_identifier(item):
        return

    n_items = _n_unique(item)
    if n_items >= 2:
        add(
            "item_levels",
            "items",
            "pass",
            f"{n_items} items are observed.",
            None,
        )
    else:
        add(
            "item_levels",
            "items",
            "fail",
            "At least two items must be observed when an item is declared.",
            n_items,
        )

    if participant_col not in data.columns or not _supported_identifier(
        data[participant_col]
    ):
        add(
            "item_crossing",
            "items",
            "fail",
            "Item crossing cannot be evaluated without participants.",
            None,
        )
        return

    frame = cast(pd.DataFrame, data.loc[:, [participant_col, item_col]]).dropna()
    if frame.empty:
        weak_items = 0
        weak_participants = 0
    else:
        participants_per_item = frame.groupby(item_col, dropna=True, observed=True)[
            participant_col
        ].nunique()
        items_per_participant = frame.groupby(participant_col, dropna=True, observed=True)[
            item_col
        ].nunique()
        weak_items = int((participants_per_item < 2).sum())
        weak_participants = int((items_per_participant < 2).sum())

    weak_total = weak_items + weak_participants
    if weak_total == 0:
        add(
            "item_crossing",
            "items",
            "pass",
            (
                "Items are observed across participants and participants "
                "are observed across items."
            ),
            None,
        )
    else:
        add(
            "item_crossing",
            "items",
            "warn",
            (
                f"{weak_items} items occur for fewer than two participants and "
                f"{weak_participants} participants occur for fewer than two items."
            ),
            weak_total,
        )


def _audit_trial_structure(
    data: pd.DataFrame,
    participant_col: str,
    trial_col: str | None,
    add: AddCheck,
) -> None:
    if trial_col is None:
        add(
            "trial_key",
            "trials",
            "pass",
            "No trial identifier was declared.",
            None,
        )
        return
    if participant_col not in data.columns or trial_col not in data.columns:
        return
    participant = data[participant_col]
    trial = data[trial_col]
    if not _supported_identifier(participant) or not _supported_identifier(trial):
        return

    frame = cast(pd.DataFrame, data.loc[:, [participant_col, trial_col]]).dropna()
    duplicated = frame.duplicated([participant_col, trial_col], keep=False)
    n_duplicated = int(duplicated.sum())
    if n_duplicated:
        add(
            "trial_key",
            "trials",
            "fail",
            f"{n_duplicated} rows have duplicated participant-trial identifiers.",
            n_duplicated,
        )
    else:
        add(
            "trial_key",
            "trials",
            "pass",
            "Participant-trial identifiers are unique.",
            None,
        )


def _audit_condition_structure(
    data: pd.DataFrame,
    participant_col: str,
    condition_col: str | None,
    random_slope: bool,
    add: AddCheck,
) -> None:
    if condition_col is None:
        add(
            "condition_levels",
            "condition",
            "pass",
            "No focal condition was declared.",
            None,
        )
        return
    if condition_col not in data.columns:
        return
    condition = data[condition_col]
    if not _supported_predictor(condition):
        add(
            "condition_type",
            "condition",
            "fail",
            "The focal condition uses an unsupported column type.",
            len(condition),
        )
        return

    add(
        "condition_type",
        "condition",
        "pass",
        "The focal condition uses a supported vector type.",
        None,
    )
    n_conditions = _n_unique(condition)
    if n_conditions >= 2:
        add(
            "condition_levels",
            "condition",
            "pass",
            f"{n_conditions} condition levels are observed.",
            None,
        )
    else:
        add(
            "condition_levels",
            "condition",
            "fail",
            "At least two condition levels must be observed.",
            n_conditions,
        )

    if not random_slope:
        add(
            "random_slope_support",
            "random_effects",
            "pass",
            "No participant-level random slope was requested.",
            None,
        )
        return

    if participant_col not in data.columns or not _supported_identifier(
        data[participant_col]
    ):
        add(
            "random_slope_support",
            "random_effects",
            "fail",
            (
                "Random-slope support cannot be evaluated without a "
                "valid participant identifier."
            ),
            None,
        )
        return

    frame = cast(pd.DataFrame, data.loc[:, [participant_col, condition_col]]).dropna()
    levels_by_participant = frame.groupby(participant_col, dropna=True, observed=True)[
        condition_col
    ].nunique()
    insufficient = int((levels_by_participant < 2).sum())
    if insufficient == 0:
        add(
            "random_slope_support",
            "random_effects",
            "pass",
            "Every participant is observed in at least two condition levels.",
            None,
        )
    else:
        add(
            "random_slope_support",
            "random_effects",
            "fail",
            f"{insufficient} participants lack within-participant condition variation.",
            insufficient,
        )

    cell_counts = frame.groupby(
        [participant_col, condition_col], dropna=True, observed=True
    ).size()
    weak_cells = int((cell_counts < 2).sum())
    if weak_cells == 0:
        add(
            "random_slope_replication",
            "random_effects",
            "pass",
            "Every observed participant-condition cell contains at least two rows.",
            None,
        )
    else:
        add(
            "random_slope_replication",
            "random_effects",
            "warn",
            f"{weak_cells} participant-condition cells contain fewer than two rows.",
            weak_cells,
        )


def _audit_time_structure(
    data: pd.DataFrame,
    participant_col: str,
    time_col: str | None,
    add: AddCheck,
) -> None:
    if time_col is None:
        add(
            "time_structure",
            "time",
            "pass",
            "No linear time or trial-order term was declared.",
            None,
        )
        return
    if time_col not in data.columns:
        return

    time = data[time_col]
    if (
        not ptypes.is_numeric_dtype(time.dtype)
        or ptypes.is_complex_dtype(time.dtype)
        or ptypes.is_bool_dtype(time.dtype)
    ):
        add(
            "time_type",
            "time",
            "fail",
            "The declared time column must be numeric.",
            len(time),
        )
        return

    add(
        "time_type",
        "time",
        "pass",
        "The declared time column is numeric.",
        None,
    )
    numeric = pd.to_numeric(time, errors="coerce")
    observed = time.notna()
    non_finite = observed & ~np.isfinite(numeric)
    n_non_finite = int(non_finite.sum())
    if n_non_finite:
        add(
            "time_finite",
            "time",
            "fail",
            f"{n_non_finite} observed time values are not finite.",
            n_non_finite,
        )
    else:
        add(
            "time_finite",
            "time",
            "pass",
            "All observed time values are finite.",
            None,
        )

    finite_values = numeric[np.isfinite(numeric)]
    if int(finite_values.nunique()) >= 2:
        add(
            "time_variation",
            "time",
            "pass",
            "The declared time column contains variation.",
            None,
        )
    else:
        add(
            "time_variation",
            "time",
            "fail",
            "The declared time column contains no usable variation.",
            None,
        )

    if participant_col not in data.columns or not _supported_identifier(
        data[participant_col]
    ):
        add(
            "time_within_participant",
            "time",
            "fail",
            (
                "Within-participant time variation cannot be evaluated "
                "without a valid participant identifier."
            ),
            None,
        )
        return

    frame = pd.DataFrame(
        {
            participant_col: data[participant_col],
            time_col: numeric,
        }
    )
    frame = frame[
        frame[participant_col].notna()
        & frame[time_col].notna()
        & np.isfinite(frame[time_col])
    ]
    time_levels = frame.groupby(participant_col, dropna=True, observed=True)[time_col].nunique()
    no_variation = int((time_levels < 2).sum())
    if time_levels.empty or no_variation == len(time_levels):
        add(
            "time_within_participant",
            "time",
            "fail",
            "No participant has within-participant time variation.",
            no_variation,
        )
    elif no_variation > 0:
        add(
            "time_within_participant",
            "time",
            "warn",
            f"{no_variation} participants lack within-participant time variation.",
            no_variation,
        )
    else:
        add(
            "time_within_participant",
            "time",
            "pass",
            "Every participant has within-participant time variation.",
            None,
        )


def _audit_predictor_structure(
    data: pd.DataFrame, predictors: tuple[str, ...] | list[str], add: AddCheck
) -> None:
    if not predictors:
        add(
            "predictor_structure",
            "predictors",
            "pass",
            "No additional predictors were declared.",
            None,
        )
        return

    existing = [column for column in predictors if column in data.columns]
    if not existing:
        return

    supported = {
        column: _supported_predictor(data[column]) for column in existing
    }
    unsupported = [column for column in existing if not supported[column]]
    if unsupported:
        add(
            "predictor_types",
            "predictors",
            "fail",
            "Unsupported predictor column types: " + ", ".join(unsupported) + ".",
            len(unsupported),
        )
    else:
        add(
            "predictor_types",
            "predictors",
            "pass",
            "All declared predictors use supported vector types.",
            None,
        )

    usable = [column for column in existing if supported[column]]
    if not usable:
        return

    non_finite_counts: dict[str, int] = {}
    for column in usable:
        series = data[column]
        if (
            ptypes.is_numeric_dtype(series.dtype)
            and not ptypes.is_complex_dtype(series.dtype)
        ):
            numeric = pd.to_numeric(series, errors="coerce")
            count = int((series.notna() & ~np.isfinite(numeric)).sum())
            if count:
                non_finite_counts[column] = count

    if non_finite_counts:
        names = list(non_finite_counts)
        add(
            "predictor_finite",
            "predictors",
            "fail",
            "Non-finite numeric predictors: " + ", ".join(names) + ".",
            sum(non_finite_counts.values()),
        )
    else:
        add(
            "predictor_finite",
            "predictors",
            "pass",
            "Numeric predictors contain only finite observed values.",
            None,
        )

    invariant = [column for column in usable if _n_unique(data[column]) < 2]
    if invariant:
        add(
            "predictor_variation",
            "predictors",
            "fail",
            "Predictors without usable variation: " + ", ".join(invariant) + ".",
            len(invariant),
        )
    else:
        add(
            "predictor_variation",
            "predictors",
            "pass",
            "All declared predictors contain usable variation.",
            None,
        )

    blank_counts: dict[str, int] = {}
    for column in usable:
        series = data[column]
        if ptypes.is_string_dtype(series.dtype) or ptypes.is_object_dtype(
            series.dtype
        ) or isinstance(series.dtype, pd.CategoricalDtype):
            text = series.astype("string")
            count = int((text.notna() & text.str.strip().eq("")).sum())
            if count:
                blank_counts[column] = count

    if blank_counts:
        names = list(blank_counts)
        add(
            "predictor_blanks",
            "predictors",
            "fail",
            "Blank text values occur in predictors: " + ", ".join(names) + ".",
            sum(blank_counts.values()),
        )
    else:
        add(
            "predictor_blanks",
            "predictors",
            "pass",
            "Text predictors contain no blank observed values.",
            None,
        )

    unused_counts: dict[str, int] = {}
    for column in usable:
        series = data[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            observed = set(series.dropna().astype(str).tolist())
            unused = set(str(level) for level in series.cat.categories) - observed
            if unused:
                unused_counts[column] = len(unused)

    if unused_counts:
        names = list(unused_counts)
        add(
            "predictor_factor_levels",
            "predictors",
            "warn",
            "Unused factor levels occur in predictors: " + ", ".join(names) + ".",
            sum(unused_counts.values()),
        )
    else:
        add(
            "predictor_factor_levels",
            "predictors",
            "pass",
            "Declared factor predictors contain no unused levels.",
            None,
        )


def _audit_interaction_structure(
    data: pd.DataFrame, contract: ModelContract, add: AddCheck
) -> None:
    terms = contract.interaction
    if terms is None:
        add(
            "interaction_support",
            "interaction",
            "pass",
            "No interaction was requested.",
            None,
        )
        return

    missing = [column for column in terms if column not in data.columns]
    if missing:
        add(
            "interaction_support",
            "interaction",
            "fail",
            "Interaction columns are missing: " + ", ".join(missing) + ".",
            len(missing),
        )
        return

    unsupported = [
        column for column in terms if not _supported_predictor(data[column])
    ]
    if unsupported:
        add(
            "interaction_support",
            "interaction",
            "fail",
            "Unsupported interaction columns: " + ", ".join(unsupported) + ".",
            len(unsupported),
        )
        return

    invariant = [column for column in terms if _n_unique(data[column]) < 2]
    if invariant:
        add(
            "interaction_support",
            "interaction",
            "fail",
            (
                "Interaction columns without usable variation: "
                + ", ".join(invariant)
                + "."
            ),
            len(invariant),
        )
        return

    if not all(_is_categorical(data[column]) for column in terms):
        add(
            "interaction_support",
            "interaction",
            "pass",
            "Both declared interaction variables contain variation.",
            None,
        )
        return

    frame = cast(pd.DataFrame, data.loc[:, list(terms)]).dropna()
    if frame.empty:
        add(
            "interaction_support",
            "interaction",
            "fail",
            "No complete interaction combinations are observed.",
            len(data),
        )
        return

    combination_counts = frame.value_counts(sort=False)
    if len(combination_counts) < 2:
        add(
            "interaction_support",
            "interaction",
            "fail",
            "Fewer than two interaction combinations are observed.",
            len(combination_counts),
        )
    elif bool((combination_counts < 2).any()):
        weak = int((combination_counts < 2).sum())
        add(
            "interaction_support",
            "interaction",
            "warn",
            f"{weak} categorical interaction combinations contain one row.",
            weak,
        )
    else:
        add(
            "interaction_support",
            "interaction",
            "pass",
            "Categorical interaction combinations are replicated.",
            None,
        )


def audit_model_readiness(
    data: pd.DataFrame, contract: ModelContract
) -> ReadinessAudit:
    """Audit observable data requirements before model construction or fitting."""
    _validate_data(data)
    _validate_contract(contract)

    checks: list[AuditCheck] = []

    def add(
        check_id: str,
        category: str,
        status: str,
        message: str,
        n_affected: int | None = None,
    ) -> None:
        if status not in {"pass", "warn", "fail"}:
            raise GP3BayesError(
                "Internal error: unsupported readiness-check status."
            )
        checks.append(
            AuditCheck(
                check_id=check_id,
                category=category,
                status=status,
                message=message,
                n_affected=n_affected,
            )
        )

    mappings = contract.mappings
    mapped = [value for value in mappings.values() if value is not None]
    predictors = list(contract.predictors)
    analysis = list(dict.fromkeys(mapped + predictors))
    missing = [column for column in analysis if column not in data.columns]
    existing = [column for column in analysis if column in data.columns]

    if len(data) > 0:
        add("data_rows", "data", "pass", f"The data contain {len(data)} rows.")
    else:
        add("data_rows", "data", "fail", "The data contain no rows.", 0)

    if not missing:
        add(
            "required_columns",
            "columns",
            "pass",
            "All declared analysis columns are present.",
        )
    else:
        add(
            "required_columns",
            "columns",
            "fail",
            "Missing declared analysis columns: " + ", ".join(missing) + ".",
            len(missing),
        )

    _audit_identifier_types(data, mappings, add)

    if not existing:
        add(
            "analysis_missingness",
            "missingness",
            "fail",
            "No declared analysis columns are available for review.",
            len(data),
        )
    else:
        missing_frame = cast(pd.DataFrame, data.loc[:, existing])
        missing_mask = missing_frame.isna().to_numpy(dtype=bool)
        n_missing_rows = int(missing_mask.any(axis=1).sum())
        if n_missing_rows:
            add(
                "analysis_missingness",
                "missingness",
                "fail",
                (
                    f"{n_missing_rows} rows contain missing values in declared "
                    "analysis columns."
                ),
                n_missing_rows,
            )
        else:
            add(
                "analysis_missingness",
                "missingness",
                "pass",
                "Declared analysis columns contain no missing values.",
            )

    outcome_col = mappings["outcome"]
    participant_col = mappings["participant"]
    if outcome_col is None or participant_col is None:
        raise GP3BayesError(
            "`contract$mappings` must contain non-null outcome and participant mappings."
        )

    if outcome_col in data.columns:
        outcome_series = cast(pd.Series, data[outcome_col])
        if contract.family == "binary":
            _audit_binary_outcome(outcome_series, add)
        else:
            _audit_duration_outcome(outcome_series, add)

    _audit_participant_structure(data, participant_col, add)
    _audit_item_structure(data, participant_col, mappings["item"], add)
    _audit_trial_structure(data, participant_col, mappings["trial"], add)
    _audit_condition_structure(
        data,
        participant_col,
        mappings["condition"],
        contract.random_slope,
        add,
    )
    _audit_time_structure(data, participant_col, mappings["time"], add)
    _audit_predictor_structure(data, predictors, add)
    _audit_interaction_structure(data, contract, add)

    checks_frame = pd.DataFrame(
        [asdict(check) for check in checks],
        columns=["check_id", "category", "status", "message", "n_affected"],
    )
    status_counts = {
        status: int((checks_frame["status"] == status).sum())
        for status in ("pass", "warn", "fail")
    }
    ready = status_counts["fail"] == 0
    if not ready:
        status = "not_ready"
    elif status_counts["warn"] > 0:
        status = "ready_with_warnings"
    else:
        status = "ready"

    return ReadinessAudit(
        audit_version="0.1",
        family=contract.family,
        model_family=contract.model_family,
        ready=ready,
        status=status,
        n_rows=len(data),
        n_columns=len(data.columns),
        status_counts=status_counts,
        checks=checks_frame,
        columns={
            "mapped": mapped,
            "predictors": predictors,
            "analysis": analysis,
            "missing": missing,
        },
        observed={
            "participants": _observed_levels(data, mappings["participant"]),
            "items": _observed_levels(data, mappings["item"]),
            "conditions": _observed_levels(data, mappings["condition"]),
        },
        contract=contract,
    )
