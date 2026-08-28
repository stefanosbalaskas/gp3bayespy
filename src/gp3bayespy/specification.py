"""Backend-independent formula and prior specifications.

This module ports the contract-to-specification layer from the frozen
``gp3bayes`` 0.5.0 reference.  It creates inspectable model declarations only;
no backend translation, sampling, or adequacy claim is performed here.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from .contracts import ModelContract, _match_contract_family
from .exceptions import GP3BayesError
from .readiness import ReadinessAudit

_R_RESERVED_NAMES = {
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

_PRIOR_COLUMNS = [
    "parameter_class",
    "distribution",
    "target",
    "location",
    "scale",
    "df",
    "shape",
    "lower",
    "upper",
    "rationale",
]


def _validate_specification_contract(contract: ModelContract) -> ModelContract:
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")

    mappings = contract.mappings
    if not isinstance(mappings, Mapping):
        raise GP3BayesError("`contract$mappings` must be a list.")

    required_mappings = (
        "outcome",
        "participant",
        "item",
        "trial",
        "condition",
        "time",
    )
    missing_mappings = [name for name in required_mappings if name not in mappings]
    if missing_mappings:
        raise GP3BayesError("`contract$mappings` is missing: " + ", ".join(missing_mappings) + ".")

    _match_contract_family(contract.family)

    rationale = contract.prior_rationale
    if (
        not isinstance(rationale, tuple)
        or len(rationale) < 4
        or any(not isinstance(value, str) or not value for value in rationale)
    ):
        raise GP3BayesError(
            "`contract$prior_rationale` must contain at least four non-empty character values."
        )

    if not isinstance(contract.random_slope, bool):
        raise GP3BayesError("`contract$random_slope` must be TRUE or FALSE.")
    if contract.random_slope and mappings["condition"] is None:
        raise GP3BayesError(
            "A participant-level random slope requires `contract$mappings$condition`."
        )
    return contract


def _validate_specification_audit(audit: ReadinessAudit) -> ReadinessAudit:
    if not isinstance(audit, ReadinessAudit):
        raise GP3BayesError("`audit` must inherit from `gp3bayes_readiness_audit`.")
    if not isinstance(audit.ready, bool):
        raise GP3BayesError("`audit$ready` must be TRUE or FALSE.")
    if audit.status not in {"ready", "ready_with_warnings", "not_ready"}:
        raise GP3BayesError("`audit$status` is invalid.")
    counts = audit.status_counts
    if (
        not isinstance(counts, Mapping)
        or not all(name in counts for name in ("pass", "warn", "fail"))
        or any(
            isinstance(counts[name], bool) or not isinstance(counts[name], numbers.Real)
            for name in ("pass", "warn", "fail")
        )
    ):
        raise GP3BayesError("`audit$status_counts` is invalid.")
    return audit


def _is_r_syntactic_name(value: str) -> bool:
    if value in _R_RESERVED_NAMES or not value:
        return False
    first = value[0]
    if first == ".":
        if len(value) > 1 and value[1].isdigit():
            return False
    elif not first.isalpha():
        return False
    return all(char.isalnum() or char in "._" for char in value[1:])


def _quote_formula_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise GP3BayesError("Formula column names must be non-empty character scalars.")
    if _is_r_syntactic_name(value):
        return value
    escaped = value.replace("\\", "\\\\").replace("`", "\\`")
    return f"`{escaped}`"


def build_model_formula(contract: ModelContract) -> str:
    """Build the approved mixed-model formula as stable R-like text."""
    _validate_specification_contract(contract)
    mappings = contract.mappings

    fixed_values = [
        mappings["condition"],
        mappings["time"],
        *contract.predictors,
    ]
    fixed: list[str] = []
    for value in fixed_values:
        if value is not None and value not in fixed:
            fixed.append(value)

    terms = [_quote_formula_name(value) for value in fixed]
    if contract.interaction is not None:
        terms.append(":".join(_quote_formula_name(value) for value in contract.interaction))

    participant = _quote_formula_name(mappings["participant"])
    if contract.random_slope:
        condition = _quote_formula_name(mappings["condition"])
        terms.append(f"(1 + {condition} | {participant})")
    else:
        terms.append(f"(1 | {participant})")

    item = mappings["item"]
    if item is not None:
        terms.append(f"(1 | {_quote_formula_name(item)})")

    outcome = _quote_formula_name(mappings["outcome"])
    return f"{outcome} ~ " + " + ".join(terms)


@dataclass(frozen=True, slots=True)
class PriorSpecification:
    """Inspectible, backend-independent prior declaration."""

    prior_version: str
    family: str
    model_family: str
    outcome_unit: str | None
    random_slope: bool
    baseline: float
    transformed_baseline: float
    table: pd.DataFrame
    backend: str = "none"
    executable: bool = False

    def __repr__(self) -> str:
        lines = [
            "<gp3bayes_prior_specification>",
            f"  Family: {self.family}",
            f"  Baseline: {self.baseline:g}",
        ]
        if self.outcome_unit is not None:
            lines.append(f"  Outcome unit: {self.outcome_unit}")
        classes = ", ".join(str(value) for value in self.table["parameter_class"].tolist())
        lines.extend(
            [
                f"  Parameter classes: {classes}",
                "  Backend: none",
                "  Executable: FALSE",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ModelSpecification:
    """Governed model specification created from contract, audit, and priors."""

    specification_version: str
    family: str
    model_family: str
    formula: str
    formula_text: str
    readiness_status: str
    warning_count: int
    contract: ModelContract
    audit: ReadinessAudit
    priors: PriorSpecification
    backend: str = "none"
    fit_performed: bool = False

    def __repr__(self) -> str:
        classes = ", ".join(str(value) for value in self.priors.table["parameter_class"].tolist())
        return "\n".join(
            [
                "<gp3bayes_model_specification>",
                f"  Family: {self.family}",
                f"  Formula: {self.formula_text}",
                f"  Readiness status: {self.readiness_status}",
                f"  Readiness warnings: {self.warning_count}",
                f"  Prior classes: {classes}",
                "  Backend: none",
                "  Fit performed: FALSE",
            ]
        )


def _validate_numeric_scalar(value: object, argument: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise GP3BayesError(f"`{argument}` must be one finite numeric value.")
    return float(value)


def _validate_positive_scalar(value: object, argument: str) -> float:
    numeric = _validate_numeric_scalar(value, argument)
    if numeric <= 0:
        raise GP3BayesError(f"`{argument}` must be strictly positive.")
    return numeric


def _validate_probability_scalar(value: object, argument: str) -> float:
    numeric = _validate_numeric_scalar(value, argument)
    if numeric <= 0 or numeric >= 1:
        raise GP3BayesError(f"`{argument}` must be strictly between zero and one.")
    return numeric


def _prior_row(
    parameter_class: str,
    distribution: str,
    target: str,
    *,
    location: float = math.nan,
    scale: float = math.nan,
    df: float = math.nan,
    shape: float = math.nan,
    lower: float = math.nan,
    upper: float = math.nan,
    rationale: str,
) -> dict[str, Any]:
    return {
        "parameter_class": parameter_class,
        "distribution": distribution,
        "target": target,
        "location": float(location),
        "scale": float(scale),
        "df": float(df),
        "shape": float(shape),
        "lower": float(lower),
        "upper": float(upper),
        "rationale": rationale,
    }


def create_prior_specification(
    contract: ModelContract,
    baseline: float | None = None,
    intercept_scale: float | None = None,
    coefficient_scale: float | None = None,
    group_sd_scale: float = 1,
    residual_scale: float | None = None,
    correlation_eta: float = 2,
    student_df: float = 3,
) -> PriorSpecification:
    """Create family-appropriate priors without backend-specific objects."""
    _validate_specification_contract(contract)
    family = contract.family

    if baseline is None:
        if family == "binary":
            baseline = 0.5
        else:
            raise GP3BayesError(
                "`baseline` must be supplied for the duration family in the recorded outcome unit."
            )

    if family == "binary":
        baseline_value = _validate_probability_scalar(baseline, "baseline")
        transformed_baseline = math.log(baseline_value / (1.0 - baseline_value))
        if intercept_scale is None:
            intercept_scale = 1.5
        if coefficient_scale is None:
            coefficient_scale = 1.0
        if residual_scale is not None:
            raise GP3BayesError(
                "`residual_scale` must be NULL for the binary family because "
                "Bernoulli residual variation is not estimated as a separate "
                "parameter."
            )
    else:
        baseline_value = _validate_positive_scalar(baseline, "baseline")
        transformed_baseline = math.log(baseline_value)
        if intercept_scale is None:
            intercept_scale = 1.0
        if coefficient_scale is None:
            coefficient_scale = 0.5
        if residual_scale is None:
            residual_scale = 1.0

    intercept_value = _validate_positive_scalar(intercept_scale, "intercept_scale")
    coefficient_value = _validate_positive_scalar(coefficient_scale, "coefficient_scale")
    group_sd_value = _validate_positive_scalar(group_sd_scale, "group_sd_scale")
    eta = _validate_numeric_scalar(correlation_eta, "correlation_eta")
    if eta < 1:
        raise GP3BayesError("`correlation_eta` must be greater than or equal to one.")
    student_df_value = _validate_positive_scalar(student_df, "student_df")

    residual_value: float | None = None
    if family == "duration":
        residual_value = _validate_positive_scalar(residual_scale, "residual_scale")

    rows = [
        _prior_row(
            "Intercept",
            "normal",
            "Population-level intercept",
            location=transformed_baseline,
            scale=intercept_value,
            lower=-math.inf,
            upper=math.inf,
            rationale=contract.prior_rationale[0],
        ),
        _prior_row(
            "b",
            "normal",
            "Population-level coefficients",
            location=0.0,
            scale=coefficient_value,
            lower=-math.inf,
            upper=math.inf,
            rationale=contract.prior_rationale[1],
        ),
        _prior_row(
            "sd",
            "student_t",
            "Group-level standard deviations",
            location=0.0,
            scale=group_sd_value,
            df=student_df_value,
            lower=0.0,
            upper=math.inf,
            rationale=contract.prior_rationale[2],
        ),
    ]
    if family == "duration":
        assert residual_value is not None
        rows.append(
            _prior_row(
                "sigma",
                "student_t",
                "Residual standard deviation",
                location=0.0,
                scale=residual_value,
                df=student_df_value,
                lower=0.0,
                upper=math.inf,
                rationale=contract.prior_rationale[2],
            )
        )
    if contract.random_slope:
        rows.append(
            _prior_row(
                "cor",
                "lkj",
                "Participant intercept-slope correlation",
                shape=eta,
                lower=-1.0,
                upper=1.0,
                rationale=contract.prior_rationale[3],
            )
        )

    table = pd.DataFrame(rows, columns=_PRIOR_COLUMNS)
    for column in ("location", "scale", "df", "shape", "lower", "upper"):
        table[column] = pd.to_numeric(table[column], errors="raise")

    priors = PriorSpecification(
        prior_version="0.1",
        family=family,
        model_family=contract.model_family,
        outcome_unit=contract.outcome_unit,
        random_slope=contract.random_slope,
        baseline=baseline_value,
        transformed_baseline=transformed_baseline,
        table=table,
    )
    return validate_prior_specification(priors, contract)


def _require_nonempty_text(series: pd.Series) -> bool:
    return bool(series.map(lambda value: isinstance(value, str) and bool(value)).all())


def _numeric_values(table: pd.DataFrame, mask: pd.Series, column: str) -> np.ndarray:
    selected = cast(pd.Series, table.loc[mask, column])
    values = pd.to_numeric(selected, errors="coerce")
    return values.to_numpy(dtype=float)


def validate_prior_specification(
    priors: PriorSpecification,
    contract: ModelContract | None = None,
) -> PriorSpecification:
    """Validate the complete prior schema and optional contract compatibility."""
    if not isinstance(priors, PriorSpecification):
        raise GP3BayesError("`priors` must inherit from `gp3bayes_prior_specification`.")

    _match_contract_family(priors.family)
    if not isinstance(priors.random_slope, bool):
        raise GP3BayesError("`priors$random_slope` must be TRUE or FALSE.")
    if priors.backend != "none":
        raise GP3BayesError('`priors$backend` must be "none".')
    if priors.executable is not False:
        raise GP3BayesError("`priors$executable` must be FALSE.")

    if contract is not None:
        _validate_specification_contract(contract)
        compatibility = [
            ("family", priors.family == contract.family),
            ("model_family", priors.model_family == contract.model_family),
            ("outcome_unit", priors.outcome_unit == contract.outcome_unit),
            ("random_slope", priors.random_slope == contract.random_slope),
        ]
        incompatible = [name for name, okay in compatibility if not okay]
        if incompatible:
            raise GP3BayesError(
                "`priors` is incompatible with `contract`: " + ", ".join(incompatible) + "."
            )

    if priors.family == "binary":
        baseline = _validate_probability_scalar(priors.baseline, "priors$baseline")
        expected = math.log(baseline / (1.0 - baseline))
    else:
        baseline = _validate_positive_scalar(priors.baseline, "priors$baseline")
        expected = math.log(baseline)

    transformed = _validate_numeric_scalar(
        priors.transformed_baseline, "priors$transformed_baseline"
    )
    if not math.isclose(transformed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise GP3BayesError(
            "`priors$transformed_baseline` is inconsistent with "
            "`priors$baseline` and the model family."
        )

    table = priors.table
    if not isinstance(table, pd.DataFrame):
        raise GP3BayesError("`priors$table` must be a data frame.")
    missing_columns = [column for column in _PRIOR_COLUMNS if column not in table]
    if missing_columns:
        raise GP3BayesError(
            "`priors$table` is missing required columns: " + ", ".join(missing_columns) + "."
        )
    if table.empty:
        raise GP3BayesError("`priors$table` must contain at least one prior row.")

    parameter_classes = table["parameter_class"].tolist()
    if pd.Series(parameter_classes).duplicated().any():
        raise GP3BayesError("Prior parameter classes must be unique.")

    expected_classes = ["Intercept", "b", "sd"]
    if priors.family == "duration":
        expected_classes.append("sigma")
    if priors.random_slope:
        expected_classes.append("cor")

    missing_classes = [value for value in expected_classes if value not in parameter_classes]
    unsupported_classes = [value for value in parameter_classes if value not in expected_classes]
    if missing_classes or unsupported_classes:
        details: list[str] = []
        if missing_classes:
            details.append("missing: " + ", ".join(missing_classes))
        if unsupported_classes:
            details.append("unsupported: " + ", ".join(map(str, unsupported_classes)))
        raise GP3BayesError(
            "Prior parameter classes are incomplete or unsupported (" + "; ".join(details) + ")."
        )

    expected_distributions = {
        "Intercept": "normal",
        "b": "normal",
        "sd": "student_t",
        "sigma": "student_t",
        "cor": "lkj",
    }
    distribution_by_class = dict(
        zip(parameter_classes, table["distribution"].tolist(), strict=True)
    )
    incorrect = [
        value
        for value in expected_classes
        if distribution_by_class.get(value) != expected_distributions[value]
    ]
    if incorrect:
        raise GP3BayesError("Incorrect prior distributions for: " + ", ".join(incorrect) + ".")

    if not _require_nonempty_text(table["target"]):
        raise GP3BayesError("Every prior row must contain a non-empty target.")
    if not _require_nonempty_text(table["rationale"]):
        raise GP3BayesError("Every prior row must contain a non-empty rationale.")

    normal = table["distribution"].eq("normal")
    normal_location = _numeric_values(table, normal, "location")
    normal_scale = _numeric_values(table, normal, "scale")
    if (
        not np.isfinite(normal_location).all()
        or not np.isfinite(normal_scale).all()
        or bool((normal_scale <= 0).any())
    ):
        raise GP3BayesError(
            "Normal priors require finite locations and strictly positive finite scales."
        )

    student = table["distribution"].eq("student_t")
    student_location = _numeric_values(table, student, "location")
    student_scale = _numeric_values(table, student, "scale")
    student_df = _numeric_values(table, student, "df")
    student_lower = _numeric_values(table, student, "lower")
    if (
        not np.isfinite(student_location).all()
        or bool((student_location != 0).any())
        or not np.isfinite(student_scale).all()
        or bool((student_scale <= 0).any())
        or not np.isfinite(student_df).all()
        or bool((student_df <= 0).any())
        or bool((student_lower != 0).any())
    ):
        raise GP3BayesError(
            "Half-Student-t priors require zero locations, positive scales and "
            "degrees of freedom, and lower bounds of zero."
        )

    lkj = table["distribution"].eq("lkj")
    if bool(lkj.any()):
        shape = _numeric_values(table, lkj, "shape")
        lower = _numeric_values(table, lkj, "lower")
        upper = _numeric_values(table, lkj, "upper")
        if (
            not np.isfinite(shape).all()
            or bool((shape < 1).any())
            or bool((lower != -1).any())
            or bool((upper != 1).any())
        ):
            raise GP3BayesError(
                "LKJ priors require finite shape values of at least one and "
                "correlation bounds from -1 to 1."
            )

    return priors


def create_model_specification(
    contract: ModelContract,
    audit: ReadinessAudit,
    priors: PriorSpecification,
) -> ModelSpecification:
    """Combine a ready audit, contract, formula, and validated priors."""
    _validate_specification_contract(contract)
    _validate_specification_audit(audit)
    if audit.contract != contract:
        raise GP3BayesError("`audit$contract` must be identical to the supplied `contract`.")
    if not audit.ready:
        raise GP3BayesError(
            f"`audit` is not ready for model specification. Status: {audit.status}."
        )

    validate_prior_specification(priors, contract)
    formula = build_model_formula(contract)
    return ModelSpecification(
        specification_version="0.1",
        family=contract.family,
        model_family=contract.model_family,
        formula=formula,
        formula_text=formula,
        readiness_status=audit.status,
        warning_count=int(audit.status_counts["warn"]),
        contract=contract,
        audit=audit,
        priors=priors,
    )
