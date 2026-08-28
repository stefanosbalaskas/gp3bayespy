"""Stable family-neutral workflow API.

The wrappers in this module perform structural dispatch only.  Completion of a
workflow stage is descriptive and never constitutes a statistical adequacy
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .exceptions import GP3BayesError


def _family(x: Any) -> str | None:
    family = getattr(x, "family", None)
    if family in {"binary", "duration", "pupil"}:
        return str(family)
    name = type(x).__name__.lower()
    for candidate in ("binary", "duration", "pupil"):
        if candidate in name:
            return candidate
    return None


def _recognized(x: Any) -> bool:
    module = type(x).__module__
    return (
        module.startswith("gp3bayespy")
        or hasattr(x, "contract_version")
        or hasattr(x, "fit_performed")
    )


@dataclass(slots=True)
class ObjectValidation:
    validation_version: str
    status: str
    class_name: str
    family: str | None
    checks: pd.DataFrame
    structural_validation_only: bool = True

    def to_frame(self) -> pd.DataFrame:
        return self.checks.copy()


def validate_gp3bayes_object(
    x: Any, recursive: bool = True, strict: bool = False
) -> ObjectValidation:
    if not isinstance(recursive, bool) or not isinstance(strict, bool):
        raise GP3BayesError("`recursive` and `strict` must be TRUE or FALSE.")
    rows: list[dict[str, str]] = []

    def check(name: str, ok: bool, detail: str, review: bool = False) -> None:
        status = "pass" if ok else ("review" if review else "fail")
        rows.append({"check": name, "status": status, "detail": detail})

    check("gp3bayes_class", _recognized(x), type(x).__name__)
    family = _family(x)
    if family is not None:
        check("approved_family", family in {"binary", "duration", "pupil"}, family)

    if hasattr(x, "contract_version"):
        required: tuple[str, ...] = (
            "family",
            "model_family",
            "mappings",
            "predictors",
            "likelihood",
            "link",
        )
        missing = [name for name in required if not hasattr(x, name)]
        check("contract_fields", not missing, "complete" if not missing else ", ".join(missing))

    if hasattr(x, "data") and hasattr(x, "contract") and hasattr(x, "transformations"):
        data = x.data
        check("prepared_fields", True, "complete")
        check(
            "prepared_data",
            isinstance(data, pd.DataFrame) and len(data) > 0,
            f"{len(data)} rows" if isinstance(data, pd.DataFrame) else "not a data frame",
        )

    if hasattr(x, "priors") and hasattr(x, "formula"):
        required = ("family", "contract", "formula", "priors")
        missing = [name for name in required if not hasattr(x, name)]
        check(
            "specification_fields", not missing, "complete" if not missing else ", ".join(missing)
        )

    if hasattr(x, "backend_fit") and hasattr(x, "sampling_backend"):
        required = (
            "family",
            "specification",
            "backend_fit",
            "sampling_backend",
            "algorithm",
            "sampling",
            "fit_performed",
        )
        missing = [name for name in required if not hasattr(x, name)]
        check("fit_fields", not missing, "complete" if not missing else ", ".join(missing))
        performed = getattr(x, "fit_performed", False) is True
        check("fit_performed", performed, str(performed), review=not performed)

    if hasattr(x, "status") and "diagnostic" in type(x).__name__.lower():
        check(
            "diagnostic_status",
            getattr(x, "status", None) is not None,
            str(getattr(x, "status", "missing")),
        )

    table = getattr(x, "table", None)
    if "summary" in type(x).__name__.lower() and table is not None:
        check(
            "posterior_summary_table",
            isinstance(table, pd.DataFrame) and len(table) > 0,
            f"{len(table)} rows" if isinstance(table, pd.DataFrame) else "missing",
        )

    if recursive:
        for name in ("contract", "prepared", "specification"):
            child = getattr(x, name, None)
            if child is None or child is x or not _recognized(child):
                continue
            child_result = validate_gp3bayes_object(child, recursive=False, strict=False)
            check(f"nested_{name}", child_result.status != "fail", child_result.status)

    frame = pd.DataFrame(rows)
    status = (
        "fail"
        if (frame["status"] == "fail").any()
        else ("review" if (frame["status"] == "review").any() else "pass")
    )
    result = ObjectValidation("0.2", status, type(x).__name__, family, frame)
    if strict and status == "fail":
        failed = ", ".join(frame.loc[frame["status"] == "fail", "check"])
        raise GP3BayesError(f"gp3bayes object validation failed: {failed}.")
    return result


def diagnose_model_fit(
    fit: Any,
    rhat_pass: float = 1.01,
    rhat_fail: float = 1.05,
    ess_per_chain_pass: float = 100,
    ess_per_chain_fail: float = 50,
    maximum_treedepth_fraction: float = 0.01,
    ebfmi_pass: float = 0.3,
    ebfmi_fail: float = 0.2,
):
    validate_gp3bayes_object(fit, strict=True)
    family = _family(fit)
    common = dict(
        rhat_pass=rhat_pass,
        rhat_fail=rhat_fail,
        ess_per_chain_pass=ess_per_chain_pass,
        ess_per_chain_fail=ess_per_chain_fail,
        maximum_treedepth_fraction=maximum_treedepth_fraction,
        ebfmi_pass=ebfmi_pass,
        ebfmi_fail=ebfmi_fail,
    )
    if family == "binary":
        from .binary import diagnose_binary_fit

        return diagnose_binary_fit(fit, **common)
    if family == "duration":
        from .duration import diagnose_duration_fit

        return diagnose_duration_fit(fit, **common)
    if family == "pupil":
        try:
            from .pupil import diagnose_pupil_fit
        except (ImportError, AttributeError) as exc:
            raise GP3BayesError("Pupil diagnostics are not available in this build.") from exc
        return diagnose_pupil_fit(fit)
    raise GP3BayesError("Unsupported gp3bayes fit family.")


def summarise_model_posterior(
    fit: Any,
    probability: float = 0.95,
    variables: Any = None,
):
    validate_gp3bayes_object(fit, strict=True)
    family = _family(fit)
    if family == "binary":
        from .binary import summarise_binary_posterior

        return summarise_binary_posterior(fit, probability=probability, variables=variables)
    if family == "duration":
        from .duration import summarise_duration_posterior

        return summarise_duration_posterior(fit, probability=probability, variables=variables)
    if family == "pupil":
        try:
            from .pupil import summarise_pupil_posterior
        except (ImportError, AttributeError) as exc:
            raise GP3BayesError(
                "Pupil posterior summaries are not available in this build."
            ) from exc
        return summarise_pupil_posterior(fit, probability=probability)
    raise GP3BayesError("Unsupported gp3bayes fit family.")


def check_model_ppc(
    fit: Any,
    draws: int = 500,
    seed: int = 1,
    pass_probability: float = 0.80,
    review_probability: float = 0.95,
):
    validate_gp3bayes_object(fit, strict=True)
    family = _family(fit)
    if family == "binary":
        from .binary import check_binary_posterior_predictive

        return check_binary_posterior_predictive(
            fit,
            draws=draws,
            seed=seed,
            pass_probability=pass_probability,
            review_probability=review_probability,
        )
    if family == "duration":
        from .duration import check_duration_posterior_predictive

        return check_duration_posterior_predictive(
            fit,
            draws=draws,
            seed=seed,
            pass_probability=pass_probability,
            review_probability=review_probability,
        )
    if family == "pupil":
        try:
            from .pupil import check_pupil_posterior_predictive
        except (ImportError, AttributeError) as exc:
            raise GP3BayesError(
                "Pupil posterior predictive checks are not available in this build."
            ) from exc
        return check_pupil_posterior_predictive(fit, ndraws=draws)
    raise GP3BayesError("Unsupported gp3bayes fit family.")


def estimate_model_estimands(fit: Any, probability: float = 0.95):
    validate_gp3bayes_object(fit, strict=True)
    family = _family(fit)
    if family == "binary":
        try:
            from .specification_closure import estimate_standardized_probability_contrast
        except (ImportError, AttributeError) as exc:
            raise GP3BayesError(
                "Binary standardized estimands are not available in this build."
            ) from exc
        return estimate_standardized_probability_contrast(fit)
    if family == "duration":
        try:
            from .specification_closure import estimate_standardized_duration_estimands
        except (ImportError, AttributeError) as exc:
            raise GP3BayesError(
                "Duration standardized estimands are not available in this build."
            ) from exc
        return estimate_standardized_duration_estimands(fit)
    if family == "pupil":
        raise GP3BayesError(
            "The frozen unified estimand dispatcher supports binary and duration fits only. "
            "Use an explicit pupil estimand function for pupil fits."
        )
    raise GP3BayesError("Unsupported gp3bayes fit family.")


def model_workflow_status(x: Any) -> pd.DataFrame:
    source_fit = x if getattr(x, "fit_performed", False) is True else getattr(x, "fit", None)
    specification = (
        x if hasattr(x, "priors") and hasattr(x, "formula") else getattr(x, "specification", None)
    )
    if specification is None and source_fit is not None:
        specification = getattr(source_fit, "specification", None)
    prepared = (
        x
        if hasattr(x, "data") and hasattr(x, "transformations")
        else getattr(specification, "prepared", None)
    )
    contract = (
        x
        if hasattr(x, "contract_version")
        else getattr(specification, "contract", None) or getattr(prepared, "contract", None)
    )
    components = getattr(x, "components", {}) or {}
    stages = pd.DataFrame(
        {
            "stage": [
                "contract",
                "prepared_data",
                "specification",
                "fit",
                "diagnostics",
                "posterior_summary",
                "ppc",
                "estimands",
                "sensitivity",
                "predictive_validation",
                "manifest",
            ],
            "completed": [
                contract is not None,
                prepared is not None,
                specification is not None,
                source_fit is not None or getattr(x, "fit_performed", False) is True,
                components.get("diagnostics") is not None if hasattr(components, "get") else False,
                components.get("posterior") is not None if hasattr(components, "get") else False,
                components.get("ppc") is not None if hasattr(components, "get") else False,
                components.get("estimands") is not None if hasattr(components, "get") else False,
                components.get("sensitivity") is not None if hasattr(components, "get") else False,
                (components.get("loo") is not None or components.get("kfold") is not None)
                if hasattr(components, "get")
                else False,
                components.get("manifest") is not None
                if hasattr(components, "get")
                else "manifest" in type(x).__name__.lower(),
            ],
        }
    )
    stages.attrs["structural_stage_map_only"] = True
    return stages


__all__ = [
    "ObjectValidation",
    "validate_gp3bayes_object",
    "diagnose_model_fit",
    "summarise_model_posterior",
    "check_model_ppc",
    "estimate_model_estimands",
    "model_workflow_status",
]
