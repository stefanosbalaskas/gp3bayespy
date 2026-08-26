"""Backend-independent model-readiness audits.

The initial Python implementation preserves the blocking checks required for
contract/specification workflows. The frozen parity ledger marks this layer
``implemented_initial`` until every R 0.5.0 warning-level edge case is ported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

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
                lines.append(f"    [{row.status.upper()}] {row.check_id}: {row.message}")
        return "\n".join(lines)


def _nlevels(data: pd.DataFrame, column: str | None) -> int | None:
    if column is None or column not in data.columns:
        return None
    return int(data[column].dropna().nunique())


def audit_model_readiness(data: pd.DataFrame, contract: ModelContract) -> ReadinessAudit:
    """Audit observable data requirements before model construction or fitting."""
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")
    if any(not isinstance(column, str) or not column for column in data.columns):
        raise GP3BayesError("`data` must have non-empty column names.")
    if data.columns.duplicated().any():
        raise GP3BayesError("`data` must not contain duplicated column names.")

    checks: list[AuditCheck] = []

    def add(
        check_id: str,
        category: str,
        status: str,
        message: str,
        n_affected: int | None = None,
    ) -> None:
        checks.append(AuditCheck(check_id, category, status, message, n_affected))

    mapped = [value for value in contract.mappings.values() if value is not None]
    analysis = list(dict.fromkeys(mapped + list(contract.predictors)))
    missing = [column for column in analysis if column not in data.columns]
    existing = [column for column in analysis if column in data.columns]

    if len(data):
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

    identifier_columns = [
        contract.mappings[key]
        for key in ("participant", "item", "trial")
        if contract.mappings[key] is not None
    ]
    available_identifiers = [
        column for column in identifier_columns if column in data.columns
    ]
    if available_identifiers:
        add(
            "identifier_types",
            "identifiers",
            "pass",
            "Declared identifier columns use supported vector types.",
        )
    else:
        add(
            "identifier_types",
            "identifiers",
            "fail",
            "No declared identifier columns are available.",
            0,
        )

    if existing:
        missing_frame = cast(pd.DataFrame, data.loc[:, existing])
        missing_mask = missing_frame.isna().to_numpy(dtype=bool)
        n_missing_rows = int(missing_mask.any(axis=1).sum())
        if n_missing_rows > 0:
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
    else:
        add(
            "analysis_missingness",
            "missingness",
            "fail",
            "No declared analysis columns are available for review.",
            len(data),
        )

    outcome = contract.mappings["outcome"]
    if outcome in data.columns:
        values = data[outcome]
        if contract.family == "binary":
            numeric = pd.to_numeric(values, errors="coerce")
            type_ok = pd.api.types.is_bool_dtype(
                values.dtype
            ) or pd.api.types.is_numeric_dtype(values.dtype)
            add(
                "outcome_type",
                "outcome",
                "pass" if type_ok else "fail",
                (
                    "The binary outcome uses a supported vector type."
                    if type_ok
                    else "The binary outcome must be a logical or numeric zero-one vector."
                ),
                None if type_ok else len(values),
            )

            invalid = values.notna() & (
                ~np.isfinite(numeric) | ~numeric.isin([0, 1])
            )
            n_invalid = int(invalid.sum())
            add(
                "outcome_values",
                "outcome",
                "fail" if invalid.any() else "pass",
                (
                    f"{n_invalid} binary outcome values are not finite zero-one values."
                    if invalid.any()
                    else "All observed binary outcome values are zero or one."
                ),
                n_invalid if invalid.any() else None,
            )

            support = sorted(set(numeric[values.notna() & ~invalid].tolist()))
            support_ok = support == [0.0, 1.0]
            add(
                "outcome_support",
                "outcome",
                "pass" if support_ok else "fail",
                (
                    "Both binary outcome classes are observed."
                    if support_ok
                    else "Both zero and one must be observed in the binary outcome."
                ),
                None if support_ok else len(support),
            )
        else:
            type_ok = pd.api.types.is_numeric_dtype(values.dtype)
            add(
                "outcome_type",
                "outcome",
                "pass" if type_ok else "fail",
                (
                    "The duration outcome uses a supported numeric type."
                    if type_ok
                    else "The duration outcome must be a numeric vector."
                ),
                None if type_ok else len(values),
            )

            numeric = pd.to_numeric(values, errors="coerce")
            nonfinite = values.notna() & ~np.isfinite(numeric)
            n_nonfinite = int(nonfinite.sum())
            add(
                "duration_finite",
                "outcome",
                "fail" if nonfinite.any() else "pass",
                (
                    f"{n_nonfinite} observed durations are not finite."
                    if nonfinite.any()
                    else "All observed durations are finite."
                ),
                n_nonfinite if nonfinite.any() else None,
            )

            nonpositive = values.notna() & np.isfinite(numeric) & (numeric <= 0)
            n_nonpositive = int(nonpositive.sum())
            add(
                "duration_positive",
                "outcome",
                "fail" if nonpositive.any() else "pass",
                (
                    f"{n_nonpositive} observed durations are not strictly positive."
                    if nonpositive.any()
                    else "All observed durations are strictly positive."
                ),
                n_nonpositive if nonpositive.any() else None,
            )

            valid = numeric[values.notna() & np.isfinite(numeric) & (numeric > 0)]
            n_unique = int(valid.nunique())
            support_ok = n_unique >= 2
            add(
                "outcome_support",
                "outcome",
                "pass" if support_ok else "fail",
                (
                    "The duration outcome contains observable variation."
                    if support_ok
                    else "The duration outcome must contain at least two values."
                ),
                None if support_ok else n_unique,
            )

    participant = contract.mappings["participant"]
    if participant in data.columns:
        n_participants = int(data[participant].dropna().nunique())
        if n_participants >= 2:
            add(
                "participant_levels",
                "participants",
                "pass",
                "At least two participants are observed.",
            )
        else:
            add(
                "participant_levels",
                "participants",
                "fail",
                "At least two participants must be observed.",
                n_participants,
            )

        participant_counts = data[participant].value_counts(dropna=True)
        if len(participant_counts) and (participant_counts >= 2).all():
            add(
                "participant_repeated",
                "participants",
                "pass",
                "Every participant has repeated observations.",
            )
        else:
            n_singletons = int((participant_counts < 2).sum())
            add(
                "participant_repeated",
                "participants",
                "warn",
                "At least one participant has fewer than two observations.",
                n_singletons,
            )

    item = contract.mappings["item"]
    if item is None:
        add("item_structure", "items", "pass", "No item identifier was declared.")
    elif item in data.columns:
        n_items = int(data[item].dropna().nunique())
        add(
            "item_levels",
            "items",
            "pass" if n_items >= 2 else "warn",
            (
                "At least two item levels are observed."
                if n_items >= 2
                else "Fewer than two item levels are observed."
            ),
            None if n_items >= 2 else n_items,
        )

    trial = contract.mappings["trial"]
    if trial is None:
        add("trial_structure", "trials", "pass", "No trial identifier was declared.")
    elif trial in data.columns and participant in data.columns:
        duplicated = data.duplicated([participant, trial], keep=False)
        n_duplicated = int(duplicated.sum())
        add(
            "trial_keys",
            "trials",
            "fail" if duplicated.any() else "pass",
            (
                f"{n_duplicated} rows share a participant-trial key."
                if duplicated.any()
                else "Participant-trial keys are unique."
            ),
            n_duplicated if duplicated.any() else None,
        )

    condition = contract.mappings["condition"]
    if condition is None:
        add(
            "condition_structure",
            "condition",
            "pass",
            "No focal condition was declared.",
        )
    elif condition in data.columns:
        n_conditions = int(data[condition].dropna().nunique())
        condition_ok = n_conditions == 2
        add(
            "condition_levels",
            "condition",
            "pass" if condition_ok else "fail",
            (
                "Exactly two focal condition levels are observed."
                if condition_ok
                else "The focal condition must have exactly two observed levels."
            ),
            None if condition_ok else n_conditions,
        )
        if contract.random_slope and participant in data.columns and condition_ok:
            by_participant = data.groupby(participant, dropna=True)[condition].nunique(
                dropna=True
            )
            unsupported = int((by_participant < 2).sum())
            add(
                "condition_random_slope_support",
                "condition",
                "pass" if unsupported == 0 else "fail",
                (
                    "Every participant contributes both condition levels for the "
                    "requested random slope."
                    if unsupported == 0
                    else (
                        f"{unsupported} participants do not contribute both condition "
                        "levels for the requested random slope."
                    )
                ),
                None if unsupported == 0 else unsupported,
            )

    time = contract.mappings["time"]
    if time is None:
        add("time_structure", "time", "pass", "No linear time variable was declared.")
    elif time in data.columns:
        numeric_time = pd.to_numeric(data[time], errors="coerce")
        invalid_time = data[time].notna() & ~np.isfinite(numeric_time)
        n_invalid_time = int(invalid_time.sum())
        add(
            "time_finite",
            "time",
            "fail" if invalid_time.any() else "pass",
            (
                f"{n_invalid_time} observed time values are non-finite."
                if invalid_time.any()
                else "Observed time values are finite."
            ),
            n_invalid_time if invalid_time.any() else None,
        )
        n_time_values = int(numeric_time.dropna().nunique())
        time_varies = n_time_values >= 2
        add(
            "time_variation",
            "time",
            "pass" if time_varies else "fail",
            (
                "The declared time variable contains variation."
                if time_varies
                else "The declared time variable lacks usable variation."
            ),
            None if time_varies else n_time_values,
        )

    if not contract.predictors:
        add(
            "predictor_structure",
            "predictors",
            "pass",
            "No additional predictors were declared.",
        )
    else:
        available_predictors = [
            column for column in contract.predictors if column in data.columns
        ]
        if available_predictors:
            add(
                "predictor_types",
                "predictors",
                "pass",
                "All declared predictors use supported vector types.",
            )
            invariant = [
                column
                for column in available_predictors
                if data[column].dropna().nunique() < 2
            ]
            add(
                "predictor_variation",
                "predictors",
                "fail" if invariant else "pass",
                (
                    "Predictors without usable variation: " + ", ".join(invariant) + "."
                    if invariant
                    else "All declared predictors contain usable variation."
                ),
                len(invariant) if invariant else None,
            )

    if contract.interaction is None:
        add(
            "interaction_support",
            "interaction",
            "pass",
            "No interaction was requested.",
        )
    else:
        missing_interaction = [
            column for column in contract.interaction if column not in data.columns
        ]
        if missing_interaction:
            add(
                "interaction_support",
                "interaction",
                "fail",
                "Interaction columns are missing: "
                + ", ".join(missing_interaction)
                + ".",
                len(missing_interaction),
            )
        elif all(
            data[column].dropna().nunique() >= 2 for column in contract.interaction
        ):
            add(
                "interaction_support",
                "interaction",
                "pass",
                "Both declared interaction variables contain variation.",
            )
        else:
            add(
                "interaction_support",
                "interaction",
                "fail",
                "Interaction columns lack usable variation.",
            )

    frame = pd.DataFrame([asdict(check) for check in checks])
    status_counts = {
        status: int((frame["status"] == status).sum())
        for status in ("pass", "warn", "fail")
    }
    ready = status_counts["fail"] == 0
    status = (
        "not_ready"
        if not ready
        else "ready_with_warnings"
        if status_counts["warn"]
        else "ready"
    )

    return ReadinessAudit(
        audit_version="0.1",
        family=contract.family,
        model_family=contract.model_family,
        ready=ready,
        status=status,
        n_rows=len(data),
        n_columns=len(data.columns),
        status_counts=status_counts,
        checks=frame,
        columns={
            "mapped": mapped,
            "predictors": list(contract.predictors),
            "analysis": analysis,
            "missing": missing,
        },
        observed={
            "participants": _nlevels(data, participant),
            "items": _nlevels(data, item),
            "conditions": _nlevels(data, condition),
        },
        contract=contract,
    )
