"""Backend-independent formula and prior specifications."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from math import log
from typing import Any

import pandas as pd

from .contracts import ModelContract
from .exceptions import GP3BayesError
from .readiness import ReadinessAudit


def _quote(name: str) -> str:
    """Mirror R backtick quoting only when syntactically necessary."""
    if re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", name) and not re.match(r"^\.[0-9]", name):
        return name
    escaped = name.replace("`", "\\`")
    return f"`{escaped}`"


def build_model_formula(contract: ModelContract) -> str:
    """Build the approved mixed-model formula as stable text."""
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")

    mappings = contract.mappings
    fixed = list(
        dict.fromkeys(
            value
            for value in (
                mappings.get("condition"),
                mappings.get("time"),
                *contract.predictors,
            )
            if value
        )
    )
    terms = [_quote(value) for value in fixed]

    if contract.interaction is not None:
        terms.append(":".join(_quote(value) for value in contract.interaction))

    participant = _quote(str(mappings["participant"]))
    if contract.random_slope:
        terms.append(f"(1 + {_quote(str(mappings['condition']))} | {participant})")
    else:
        terms.append(f"(1 | {participant})")

    if mappings.get("item") is not None:
        terms.append(f"(1 | {_quote(str(mappings['item']))})")

    return f"{_quote(str(mappings['outcome']))} ~ " + " + ".join(terms)


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


def _positive(value: Any, name: str) -> float:
    message = f"`{name}` must be a positive finite numeric scalar."
    if isinstance(value, bool):
        raise GP3BayesError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(message) from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise GP3BayesError(message)
    return numeric


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
    """Create family-appropriate priors without creating backend objects."""
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")

    family = contract.family
    if baseline is None:
        if family == "binary":
            baseline = 0.5
        else:
            raise GP3BayesError(
                "`baseline` must be supplied for the duration family in the recorded "
                "outcome unit."
            )

    if family == "binary":
        binary_baseline = float(baseline)
        if not 0 < binary_baseline < 1:
            raise GP3BayesError("`baseline` must be strictly between zero and one.")
        transformed_baseline = log(binary_baseline / (1 - binary_baseline))
        intercept_scale = 1.5 if intercept_scale is None else intercept_scale
        coefficient_scale = 1 if coefficient_scale is None else coefficient_scale
        if residual_scale is not None:
            raise GP3BayesError(
                "`residual_scale` must be NULL for the binary family because Bernoulli "
                "residual variation is not estimated as a separate parameter."
            )
        baseline_value = binary_baseline
    else:
        duration_baseline = _positive(baseline, "baseline")
        transformed_baseline = log(duration_baseline)
        intercept_scale = 1 if intercept_scale is None else intercept_scale
        coefficient_scale = 0.5 if coefficient_scale is None else coefficient_scale
        residual_scale = 1 if residual_scale is None else residual_scale
        baseline_value = duration_baseline

    intercept_scale_value = _positive(intercept_scale, "intercept_scale")
    coefficient_scale_value = _positive(coefficient_scale, "coefficient_scale")
    group_sd_scale_value = _positive(group_sd_scale, "group_sd_scale")
    student_df_value = _positive(student_df, "student_df")

    eta = float(correlation_eta)
    if not math.isfinite(eta) or eta < 1:
        raise GP3BayesError("`correlation_eta` must be greater than or equal to one.")

    rows: list[dict[str, Any]] = [
        {
            "parameter_class": "Intercept",
            "distribution": "normal",
            "target": "Population-level intercept",
            "location": transformed_baseline,
            "scale": intercept_scale_value,
            "df": None,
            "shape": None,
            "lower": float("-inf"),
            "upper": float("inf"),
            "rationale": contract.prior_rationale[0],
        },
        {
            "parameter_class": "b",
            "distribution": "normal",
            "target": "Population-level coefficients",
            "location": 0.0,
            "scale": coefficient_scale_value,
            "df": None,
            "shape": None,
            "lower": float("-inf"),
            "upper": float("inf"),
            "rationale": contract.prior_rationale[1],
        },
        {
            "parameter_class": "sd",
            "distribution": "student_t",
            "target": "Group-level standard deviations",
            "location": 0.0,
            "scale": group_sd_scale_value,
            "df": student_df_value,
            "shape": None,
            "lower": 0.0,
            "upper": float("inf"),
            "rationale": contract.prior_rationale[2],
        },
    ]

    if family == "duration":
        rows.append(
            {
                "parameter_class": "sigma",
                "distribution": "student_t",
                "target": "Residual standard deviation",
                "location": 0.0,
                "scale": _positive(residual_scale, "residual_scale"),
                "df": student_df_value,
                "shape": None,
                "lower": 0.0,
                "upper": float("inf"),
                "rationale": contract.prior_rationale[2],
            }
        )

    if contract.random_slope:
        rows.append(
            {
                "parameter_class": "cor",
                "distribution": "lkj",
                "target": "Participant intercept-slope correlation",
                "location": None,
                "scale": None,
                "df": None,
                "shape": eta,
                "lower": -1.0,
                "upper": 1.0,
                "rationale": contract.prior_rationale[3],
            }
        )

    priors = PriorSpecification(
        prior_version="0.1",
        family=family,
        model_family=contract.model_family,
        outcome_unit=contract.outcome_unit,
        random_slope=contract.random_slope,
        baseline=float(baseline_value),
        transformed_baseline=transformed_baseline,
        table=pd.DataFrame(rows),
    )
    validate_prior_specification(priors, contract)
    return priors


def validate_prior_specification(
    priors: PriorSpecification,
    contract: ModelContract | None = None,
) -> PriorSpecification:
    """Validate required prior classes and optional contract compatibility."""
    if not isinstance(priors, PriorSpecification):
        raise GP3BayesError("`priors` must inherit from `gp3bayes_prior_specification`.")

    required = {"Intercept", "b", "sd"}
    if priors.family == "duration":
        required.add("sigma")
    if priors.random_slope:
        required.add("cor")

    actual = set(priors.table["parameter_class"].astype(str))
    if not required.issubset(actual):
        raise GP3BayesError("`priors$table` is missing required parameter classes.")

    incompatible = contract is not None and (
        priors.family != contract.family
        or priors.model_family != contract.model_family
        or priors.random_slope != contract.random_slope
        or priors.outcome_unit != contract.outcome_unit
    )
    if incompatible:
        raise GP3BayesError("`priors` is not compatible with the supplied `contract`.")

    return priors


def create_model_specification(
    contract: ModelContract,
    audit: ReadinessAudit,
    priors: PriorSpecification,
) -> ModelSpecification:
    """Combine a ready audit, contract, formula, and validated priors."""
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must inherit from `gp3bayes_model_contract`.")
    if not isinstance(audit, ReadinessAudit):
        raise GP3BayesError("`audit` must inherit from `gp3bayes_readiness_audit`.")
    if audit.contract != contract:
        raise GP3BayesError(
            "`audit$contract` must be identical to the supplied `contract`."
        )
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
        warning_count=audit.status_counts["warn"],
        contract=contract,
        audit=audit,
        priors=priors,
    )
