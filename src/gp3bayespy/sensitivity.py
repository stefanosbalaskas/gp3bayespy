"""Unified sensitivity, recovery, SBC, and evidence publication helpers.

Python adaptation of gp3bayes 0.5.0 ``sensitivity-evidence-suite.R`` and
``recovery-sensitivity-publication.R``.  Sensitivity components are opt-in;
plots and reports are descriptive and never convert stability into a universal
robustness, adequacy, exclusion, or model-selection claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise GP3BayesError("Matplotlib is required for sensitivity plots.") from exc
    return plt


def _family(fit: Any) -> str:
    value = getattr(fit, "family", None)
    if value not in {"binary", "duration"}:
        raise GP3BayesError("`fit` must be an approved binary or duration gp3bayes fit.")
    return str(value)


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise GP3BayesError(f"`{name}` must be TRUE or FALSE.")
    return bool(value)


def _mapping(value: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GP3BayesError(f"`{name}` must be a mapping.")
    return dict(value)


def _table_field(x: Any, *names: str) -> pd.DataFrame:
    for name in names:
        value = getattr(x, name, None)
        if value is None and isinstance(x, Mapping):
            value = x.get(name)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    raise GP3BayesError(f"The object does not expose a tabular `{names[0]}` field.")


def _status(x: Any) -> str:
    if x is None:
        return "not_run"
    if isinstance(x, SuiteError):
        return "error"
    value = getattr(x, "status", None)
    if value is None and isinstance(x, Mapping):
        value = x.get("status")
    if value is not None:
        return str(value)
    adequacy = getattr(x, "adequacy_established", None)
    if adequacy is True:
        return "pass"
    return "completed"


@dataclass(slots=True)
class SensitivityPlan:
    plan_version: str = "0.2"
    prior_scale: Mapping[str, Any] = field(default_factory=lambda: {"run": False, "args": {}})
    powerscale: Mapping[str, Any] = field(default_factory=lambda: {"run": False, "args": {}})
    psis_loo: Mapping[str, Any] = field(default_factory=lambda: {"run": False, "args": {}})
    random_slope: Mapping[str, Any] = field(default_factory=lambda: {"plan": None, "args": {}})
    group_deletion: Mapping[str, Any] = field(default_factory=lambda: {"plan": None, "args": {}})
    alternative_estimands: Mapping[str, Any] = field(default_factory=dict)
    duration_unit: Mapping[str, Any] | None = None
    automatic_model_selection: bool = False
    automatic_exclusion: bool = False


@dataclass(slots=True)
class SuiteError:
    status: str
    message: str


@dataclass(slots=True)
class SensitivitySuite:
    suite_version: str
    family: str
    status: str
    fit: Any
    plan: SensitivityPlan
    reference_estimand: Any
    results: Mapping[str, Any]
    component_status: pd.DataFrame
    robustness_established: bool = False
    automatic_model_selection: bool = False
    automatic_exclusion: bool = False
    interpretation: str = (
        "The suite collates declared sensitivity analyses. It does not convert "
        "component stability into a universal robustness claim."
    )


@dataclass(slots=True)
class PriorSensitivity:
    sensitivity_version: str
    family: str
    scale_multipliers: Mapping[str, float]
    comparison: pd.DataFrame
    scenario_status: pd.DataFrame
    reference_diagnostic_status: str
    status: str
    alternative_fits: Mapping[str, Any] | None = None
    sensitivity_assessed: bool = True
    robustness_claim: bool = False
    posterior_adequacy_established: bool = False


@dataclass(slots=True)
class RecoveryResult:
    recovery_version: str
    family: str
    parameter_summary: pd.DataFrame
    estimates: pd.DataFrame
    fit_status: pd.DataFrame
    repetitions: int
    status: str
    validation_established: bool = False
    automatic_model_selection: bool = False
    interpretation: str = (
        "Recovery thresholds summarize the declared simulation design only; "
        "a passing run is not an automatic validation claim."
    )


@dataclass(slots=True)
class ModelReport:
    family: str
    file: str
    registry: pd.DataFrame
    automatic_convergence_claim: bool = False
    posterior_adequacy_established: bool = False
    substantive_validity_established: bool = False


def _worst_status(values: Sequence[str]) -> str:
    order = {
        "pass": 0,
        "completed": 0,
        "not_assessed": 1,
        "review": 1,
        "warn": 1,
        "fail": 2,
        "error": 3,
    }
    cleaned = [str(v) for v in values if v is not None]
    if not cleaned:
        return "not_assessed"
    return max(cleaned, key=lambda value: order.get(value, 1))


def _classify_upper(value: float, pass_threshold: float, review_threshold: float) -> str:
    if not np.isfinite(value):
        return "review"
    if value <= pass_threshold:
        return "pass"
    if value <= review_threshold:
        return "review"
    return "fail"


@dataclass(slots=True)
class ModelEvidence:
    evidence_version: str
    family: str | None
    fit: Any
    components: Mapping[str, Any]
    component_table: pd.DataFrame
    adequacy_established: bool = False
    robustness_established: bool = False
    causal_identification_established: bool = False
    automatic_model_selection: bool = False
    interpretation: str = (
        "Evidence components are collected for transparent review. "
        "No aggregate pass/fail adequacy verdict is generated."
    )


def create_sensitivity_suite_plan(
    prior_scale: bool = False,
    powerscale: bool = False,
    psis_loo: bool = False,
    random_slope_plan: Any = None,
    group_deletion_plan: Any = None,
    alternative_estimands: Mapping[str, Any] | None = None,
    duration_unit: Mapping[str, Any] | None = None,
    prior_scale_args: Mapping[str, Any] | None = None,
    powerscale_args: Mapping[str, Any] | None = None,
    psis_args: Mapping[str, Any] | None = None,
    random_slope_args: Mapping[str, Any] | None = None,
    group_deletion_args: Mapping[str, Any] | None = None,
) -> SensitivityPlan:
    """Create an inert, declarative sensitivity-suite plan."""
    return SensitivityPlan(
        prior_scale={
            "run": _flag(prior_scale, "prior_scale"),
            "args": _mapping(prior_scale_args, "prior_scale_args"),
        },
        powerscale={
            "run": _flag(powerscale, "powerscale"),
            "args": _mapping(powerscale_args, "powerscale_args"),
        },
        psis_loo={"run": _flag(psis_loo, "psis_loo"), "args": _mapping(psis_args, "psis_args")},
        random_slope={
            "plan": random_slope_plan,
            "args": _mapping(random_slope_args, "random_slope_args"),
        },
        group_deletion={
            "plan": group_deletion_plan,
            "args": _mapping(group_deletion_args, "group_deletion_args"),
        },
        alternative_estimands=_mapping(alternative_estimands, "alternative_estimands"),
        duration_unit=None if duration_unit is None else _mapping(duration_unit, "duration_unit"),
    )


def _safe_call(function: Any, kwargs: Mapping[str, Any], stop_on_error: bool) -> Any:
    try:
        return function(**dict(kwargs))
    except Exception as exc:
        if stop_on_error:
            raise
        return SuiteError("error", str(exc))


def run_sensitivity_suite(
    fit: Any,
    plan: SensitivityPlan | None = None,
    reference_estimand: Any = None,
    stop_on_error: bool = False,
) -> SensitivitySuite:
    """Run only sensitivity components explicitly enabled in ``plan``."""
    family = _family(fit)
    if plan is None:
        plan = create_sensitivity_suite_plan()
    if not isinstance(plan, SensitivityPlan):
        raise GP3BayesError("`plan` must be created by `create_sensitivity_suite_plan()`.")
    stop = _flag(stop_on_error, "stop_on_error")
    results: dict[str, Any] = {}

    if bool(plan.prior_scale["run"]):
        function: Any
        if family == "binary":
            from .binary import assess_binary_prior_sensitivity as function
        else:
            from .duration import (
                assess_duration_prior_sensitivity as function,  # type: ignore[assignment]
            )
        results["prior_scale"] = _safe_call(
            function, {"fit": fit, **dict(plan.prior_scale["args"])}, stop
        )

    if bool(plan.powerscale["run"]):
        from .advanced_optional_workflows import assess_powerscaled_sensitivity

        results["powerscale"] = _safe_call(
            assess_powerscaled_sensitivity,
            {"fit": fit, **dict(plan.powerscale["args"])},
            stop,
        )

    if bool(plan.psis_loo["run"]):
        from .advanced_optional_workflows import compute_psis_loo

        results["psis_loo"] = _safe_call(
            compute_psis_loo, {"fit": fit, **dict(plan.psis_loo["args"])}, stop
        )

    if plan.random_slope["plan"] is not None:
        from .specification_closure import run_random_slope_sensitivity

        results["random_slope"] = _safe_call(
            run_random_slope_sensitivity,
            {"plan": plan.random_slope["plan"], **dict(plan.random_slope["args"])},
            stop,
        )

    if plan.group_deletion["plan"] is not None:
        from .specification_closure import run_group_deletion_sensitivity

        results["group_deletion"] = _safe_call(
            run_group_deletion_sensitivity,
            {"plan": plan.group_deletion["plan"], **dict(plan.group_deletion["args"])},
            stop,
        )

    if plan.alternative_estimands:
        from .specification_closure import compare_estimand_sensitivity

        if reference_estimand is None:
            from .unified_workflow_api import estimate_model_estimands

            try:
                reference_estimand = estimate_model_estimands(fit)
            except Exception as exc:
                if stop:
                    raise
                reference_estimand = SuiteError("error", str(exc))
        if not isinstance(reference_estimand, SuiteError):
            results["estimand_alternatives"] = _safe_call(
                compare_estimand_sensitivity,
                {"reference": reference_estimand, "alternatives": plan.alternative_estimands},
                stop,
            )

    if plan.duration_unit is not None:
        required = {"estimand", "multiplier"}
        if not required.issubset(plan.duration_unit):
            raise GP3BayesError("`duration_unit` must contain `estimand` and `multiplier`.")
        if reference_estimand is None:
            from .unified_workflow_api import estimate_model_estimands

            reference_estimand = _safe_call(estimate_model_estimands, {"fit": fit}, stop)
        from .specification_closure import audit_duration_unit_invariance

        if not isinstance(reference_estimand, SuiteError):
            results["duration_unit"] = _safe_call(
                audit_duration_unit_invariance,
                {
                    "reference": reference_estimand,
                    "converted": plan.duration_unit["estimand"],
                    "multiplier": plan.duration_unit["multiplier"],
                    "tolerance": plan.duration_unit.get("tolerance", 0.02),
                },
                stop,
            )

    statuses = {name: _status(value) for name, value in results.items()}
    if any(value in {"error", "fail"} for value in statuses.values()) or any(
        value in {"review", "warn", "not_assessed"} for value in statuses.values()
    ):
        overall = "review"
    elif statuses:
        overall = "completed"
    else:
        overall = "not_run"
    table = pd.DataFrame(
        [{"component": name, "status": value} for name, value in statuses.items()],
        columns=["component", "status"],
    )
    return SensitivitySuite("0.2", family, overall, fit, plan, reference_estimand, results, table)


def summarise_sensitivity_suite(x: SensitivitySuite) -> pd.DataFrame:
    if not isinstance(x, SensitivitySuite):
        raise GP3BayesError("`x` must be a gp3bayes sensitivity suite.")
    if x.component_status.empty:
        return pd.DataFrame(columns=["component", "status", "detail"])
    rows = []
    for row in x.component_status.itertuples(index=False):
        result = x.results[str(row.component)]
        if isinstance(result, SuiteError):
            detail = result.message
        else:
            detail = getattr(result, "interpretation", type(result).__name__)
        rows.append({"component": row.component, "status": row.status, "detail": str(detail)})
    return pd.DataFrame(rows)


def collect_model_evidence(
    fit: Any = None,
    design: Any = None,
    diagnostics: Any = None,
    posterior: Any = None,
    ppc: Any = None,
    estimands: Any = None,
    loo: Any = None,
    kfold: Any = None,
    sensitivity: Any = None,
    manifest: Any = None,
    compute: Sequence[str] = (),
) -> ModelEvidence:
    """Collect supplied evidence components without generating an adequacy verdict."""
    allowed = {"diagnostics", "posterior", "estimands"}
    requested = tuple(dict.fromkeys(str(value) for value in compute))
    if any(value not in allowed for value in requested):
        raise GP3BayesError("`compute` may contain only diagnostics, posterior, estimands.")
    if requested and fit is None:
        raise GP3BayesError("A gp3bayes `fit` is required for requested computed components.")
    if "diagnostics" in requested and diagnostics is None:
        from .unified_workflow_api import diagnose_model_fit

        diagnostics = diagnose_model_fit(fit)
    if "posterior" in requested and posterior is None:
        from .unified_workflow_api import summarise_model_posterior

        posterior = summarise_model_posterior(fit)
    if "estimands" in requested and estimands is None:
        from .unified_workflow_api import estimate_model_estimands

        estimands = estimate_model_estimands(fit)

    components = {
        "design": design,
        "diagnostics": diagnostics,
        "posterior": posterior,
        "ppc": ppc,
        "estimands": estimands,
        "loo": loo,
        "kfold": kfold,
        "sensitivity": sensitivity,
        "manifest": manifest,
    }
    rows = []
    for name, value in components.items():
        status = "not_supplied" if value is None else _status(value)
        if status == "completed":
            status = "available"
        rows.append({"component": name, "available": value is not None, "status": status})
    family = getattr(fit, "family", None)
    if family is None:
        family = getattr(manifest, "family", None) or getattr(estimands, "family", None)
    return ModelEvidence("0.2", family, fit, components, pd.DataFrame(rows))


def create_model_evidence_report(
    evidence: ModelEvidence, file: str | Path, overwrite: bool = False
) -> str:
    if not isinstance(evidence, ModelEvidence):
        raise GP3BayesError("`evidence` must be created by `collect_model_evidence()`.")
    path = Path(file)
    if not str(path):
        raise GP3BayesError("`file` must be one explicit non-empty path.")
    if path.exists() and not _flag(overwrite, "overwrite"):
        raise GP3BayesError("`file` already exists. Set `overwrite=True` to replace it.")
    if not path.parent.exists():
        raise GP3BayesError("The report parent directory does not exist.")
    lines = [
        "# gp3bayes model evidence report",
        "",
        f"Family: {evidence.family}",
        "",
        "## Evidence inventory",
        "",
    ]
    for row in evidence.component_table.itertuples(index=False):
        detail = f"available ({row.status})" if row.available else "not supplied"
        lines.append(f"- {row.component}: {detail}")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This report is an evidence inventory. It does not automatically establish "
        "convergence, posterior adequacy, robustness, causal identification, "
        "substantive validity, or a preferred model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


# Publication table adapters -------------------------------------------------


def recovery_parameter_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "parameter_summary")


def recovery_estimate_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "estimates")


def recovery_fit_status_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "fit_status")


def prior_sensitivity_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "comparison")


def prior_sensitivity_scenario_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "scenario_status")


def estimand_sensitivity_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "table")


def group_deletion_sensitivity_table(x: Any) -> pd.DataFrame:
    return _table_field(x, "summary")


def random_slope_sensitivity_table(x: Any) -> pd.DataFrame:
    comparison = getattr(x, "comparison", None)
    if comparison is None and isinstance(x, Mapping):
        comparison = x.get("comparison")
    if comparison is None:
        raise GP3BayesError("The random-slope object has no valid estimand comparison.")
    return estimand_sensitivity_table(comparison)


def powerscale_sensitivity_table(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    raw = getattr(x, "raw", None)
    if raw is None and isinstance(x, Mapping):
        raw = x.get("raw")
    if raw is None:
        # The Python priorsense adaptation returns a DataFrame directly.
        try:
            return pd.DataFrame(x).copy()
        except Exception as exc:
            raise GP3BayesError("Could not convert the power-scale sensitivity result.") from exc
    try:
        return pd.DataFrame(raw).copy()
    except Exception as exc:
        raise GP3BayesError("Could not convert the power-scale sensitivity result.") from exc


def sbc_stats_table(x: Any) -> pd.DataFrame:
    from .advanced_optional_workflows import SBCResult

    if isinstance(x, SBCResult):
        if not x.ranks.empty:
            return x.ranks.copy()
        return x.simulations.copy()
    raw = getattr(x, "raw", None)
    if raw is None and isinstance(x, Mapping):
        raw = x.get("raw")
    stats = raw.get("stats") if isinstance(raw, Mapping) else None
    if stats is None:
        raise GP3BayesError("The SBC result does not expose tabular statistics.")
    return pd.DataFrame(stats).copy()


def sbc_overview_table(x: Any) -> pd.DataFrame:
    from .advanced_optional_workflows import SBCResult

    stats = sbc_stats_table(x)
    if isinstance(x, SBCResult):
        plan = x.plan
        variable_col = (
            "parameter"
            if "parameter" in stats.columns
            else "variable"
            if "variable" in stats.columns
            else None
        )
        n_variables = stats[variable_col].astype(str).nunique() if variable_col else np.nan
        return pd.DataFrame(
            [
                {
                    "status": "completed" if not x.simulations.empty else "not_run",
                    "family": getattr(plan.specification, "family", None),
                    "backend": plan.backend,
                    "simulations": plan.n_sims,
                    "variables_recorded": n_variables,
                    "diagnostics_inspected": not x.ranks.empty,
                    "calibration_established": False,
                }
            ]
        )
    plan = getattr(x, "plan", None)  # type: ignore[assignment]
    if plan is None and isinstance(x, Mapping):
        plan = x.get("plan", {})

    def take(name: str, default: Any = None) -> Any:
        return (
            plan.get(name, default) if isinstance(plan, Mapping) else getattr(plan, name, default)
        )

    variable_col = next((name for name in ("variable", "parameter") if name in stats.columns), None)
    n_variables = stats[variable_col].astype(str).nunique() if variable_col else np.nan
    status = getattr(x, "status", None) if not isinstance(x, Mapping) else x.get("status")
    inspected = (
        getattr(x, "diagnostics_inspected", None)
        if not isinstance(x, Mapping)
        else x.get("diagnostics_inspected")
    )
    return pd.DataFrame(
        [
            {
                "status": status,
                "family": take("family"),
                "backend": take("backend"),
                "simulations": take("n_sims"),
                "variables_recorded": n_variables,
                "diagnostics_inspected": inspected,
                "calibration_established": False,
            }
        ]
    )


# Matplotlib publication adapters -------------------------------------------


def _frame(x: Any, extractor: Any) -> pd.DataFrame:
    return x.copy() if isinstance(x, pd.DataFrame) else extractor(x)


def _barh(frame: pd.DataFrame, value: str, label: str, title: str, xlabel: str):
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.barh(frame[label].astype(str), frame[value].astype(float))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return fig


def plot_recovery_bias(x: Any):
    d = _frame(x, recovery_parameter_table)
    if not {"variable", "standardized_bias"}.issubset(d.columns):
        raise GP3BayesError("Recovery summaries require variable and standardized_bias.")
    return _barh(d, "standardized_bias", "variable", "Parameter recovery bias", "Standardized bias")


def plot_recovery_coverage(x: Any):
    d = _frame(x, recovery_parameter_table)
    if not {"variable", "coverage"}.issubset(d.columns):
        raise GP3BayesError("Recovery summaries require variable and coverage.")
    return _barh(d, "coverage", "variable", "Parameter recovery coverage", "Coverage")


def plot_recovery_rmse(x: Any):
    d = _frame(x, recovery_parameter_table)
    if not {"variable", "rmse"}.issubset(d.columns):
        raise GP3BayesError("Recovery summaries require variable and rmse.")
    return _barh(d, "rmse", "variable", "Parameter recovery RMSE", "RMSE")


def plot_recovery_estimates(x: Any, variables: Sequence[str] | None = None):
    d = _frame(x, recovery_estimate_table)
    required = {"variable", "truth", "median", "lower", "upper", "repetition"}
    if not required.issubset(d.columns):
        raise GP3BayesError("Recovery estimates do not contain the required columns.")
    if variables is not None:
        d = d[d["variable"].astype(str).isin([str(v) for v in variables])]
    if d.empty:
        raise GP3BayesError("No recovery rows remain after filtering.")
    plt = _mpl()
    fig, ax = plt.subplots()
    for variable, g in d.groupby("variable", sort=False):
        ax.errorbar(
            g["repetition"],
            g["median"],
            yerr=[g["median"] - g["lower"], g["upper"] - g["median"]],
            fmt="o",
            label=str(variable),
        )
        ax.plot(g["repetition"], g["truth"], linestyle="--")
    ax.set_title("Repetition-level parameter recovery")
    ax.set_xlabel("Recovery repetition")
    ax.set_ylabel("Posterior estimate")
    ax.legend()
    return fig


def plot_recovery_fit_status(x: Any):
    d = _frame(x, recovery_fit_status_table)
    if not {"diagnostic_status", "completed"}.issubset(d.columns):
        raise GP3BayesError("Recovery fit statuses require diagnostic_status and completed.")
    tab = d.groupby(["diagnostic_status", "completed"], dropna=False).size().unstack(fill_value=0)
    plt = _mpl()
    fig, ax = plt.subplots()
    tab.plot(kind="bar", ax=ax)
    ax.set_title("Recovery-fit completion and diagnostics")
    ax.set_ylabel("Repetitions")
    return fig


def plot_prior_sensitivity(x: Any):
    d = _frame(x, prior_sensitivity_table)
    required = {"scenario", "scale_multiplier", "variable", "standardized_shift"}
    if not required.issubset(d.columns):
        raise GP3BayesError(
            "Prior sensitivity requires scenario, scale_multiplier, variable, standardized_shift."
        )
    plt = _mpl()
    fig, ax = plt.subplots()
    for variable, g in d.groupby("variable", sort=False):
        ax.plot(g["scale_multiplier"], g["standardized_shift"], marker="o", label=str(variable))
    ax.set_title("Declared prior-scale sensitivity")
    ax.set_xlabel("Prior scale multiplier")
    ax.set_ylabel("Absolute standardized posterior-median shift")
    ax.legend()
    return fig


def plot_prior_sensitivity_scenarios(x: Any):
    d = _frame(x, prior_sensitivity_scenario_table)
    if not {"scenario", "maximum_standardized_shift"}.issubset(d.columns):
        raise GP3BayesError(
            "Scenario sensitivity requires scenario and maximum_standardized_shift."
        )
    return _barh(
        d,
        "maximum_standardized_shift",
        "scenario",
        "Prior-sensitivity scenario maxima",
        "Maximum standardized shift",
    )


def plot_estimand_sensitivity_gg(x: Any):
    d = _frame(x, estimand_sensitivity_table)
    required = {
        "alternative",
        "reference_median",
        "alternative_median",
        "alternative_lower",
        "alternative_upper",
    }
    if not required.issubset(d.columns):
        raise GP3BayesError("Estimand sensitivity summaries are incomplete.")
    plt = _mpl()
    fig, ax = plt.subplots()
    y = np.arange(len(d))
    med = d["alternative_median"].to_numpy(float)
    ax.errorbar(
        med,
        y,
        xerr=[
            med - d["alternative_lower"].to_numpy(float),
            d["alternative_upper"].to_numpy(float) - med,
        ],
        fmt="o",
    )
    ax.axvline(float(d["reference_median"].iloc[0]), linestyle="--")
    ax.set_yticks(y, d["alternative"].astype(str))
    ax.set_title("Estimand sensitivity")
    ax.set_xlabel("Posterior estimand")
    return fig


def plot_group_deletion_sensitivity(x: Any):
    d = _frame(x, group_deletion_sensitivity_table)
    if not {"omitted_unit", "median_shift"}.issubset(d.columns):
        raise GP3BayesError("Group-deletion sensitivity requires omitted_unit and median_shift.")
    fig = _barh(
        d,
        "median_shift",
        "omitted_unit",
        "Declared group-deletion sensitivity",
        "Posterior median shift",
    )
    return fig


def plot_random_slope_sensitivity(x: Any):
    d = random_slope_sensitivity_table(x) if not isinstance(x, pd.DataFrame) else x
    fig = plot_estimand_sensitivity_gg(d)
    fig.axes[0].set_title("Random-intercept versus random-slope sensitivity")
    return fig


def plot_powerscale_sensitivity_gg(x: Any):
    d = _frame(x, powerscale_sensitivity_table)
    if {"variable", "prior", "likelihood"}.issubset(d.columns):
        labels = d["variable"].astype(str)
        prior = d["prior"].to_numpy(float)
        likelihood = d["likelihood"].to_numpy(float)
    elif {"component", "alpha", "distance"}.issubset(d.columns):
        labels = d["variable"].fillna("all").astype(str) + ":" + d["component"].astype(str)
        prior = np.nan_to_num(d["distance"].to_numpy(float), nan=0.0)
        likelihood = np.zeros(len(d))
    else:
        raise GP3BayesError("Power-scale sensitivity does not contain plottable columns.")
    plt = _mpl()
    fig, ax = plt.subplots()
    y = np.arange(len(d))
    ax.barh(y - 0.2, prior, height=0.4, label="prior")
    ax.barh(y + 0.2, likelihood, height=0.4, label="likelihood")
    ax.set_yticks(y, labels)
    ax.set_title("Prior and likelihood power-scale sensitivity")
    ax.legend()
    return fig


def _sbc_plot_data(x: Any, variables: Sequence[str] | None) -> pd.DataFrame:
    d = sbc_stats_table(x)
    parameter = (
        "parameter" if "parameter" in d.columns else "variable" if "variable" in d.columns else None
    )
    if variables is not None and parameter is not None:
        d = d[d[parameter].astype(str).isin([str(v) for v in variables])]
    if d.empty:
        raise GP3BayesError("No SBC statistics are available for plotting.")
    return d


def plot_sbc_rank_gg(x: Any, variables: Sequence[str] | None = None):
    d = _sbc_plot_data(x, variables)
    rank = "rank"
    max_rank = "draws" if "draws" in d.columns else "max_rank"
    if rank not in d.columns:
        raise GP3BayesError("SBC rank statistics are unavailable.")
    scale = d[max_rank].to_numpy(float) if max_rank in d.columns else np.maximum(d[rank].max(), 1)
    u = d[rank].to_numpy(float) / np.maximum(scale, 1)
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.hist(u, bins=min(10, max(3, len(u) // 2)))
    ax.set_title("SBC rank fractions")
    ax.set_xlabel("Rank fraction")
    return fig


def plot_sbc_ecdf_gg(x: Any, variables: Sequence[str] | None = None):
    d = _sbc_plot_data(x, variables)
    rank = "rank"
    max_rank = "draws" if "draws" in d.columns else "max_rank"
    if rank not in d.columns:
        raise GP3BayesError("SBC rank statistics are unavailable.")
    scale = d[max_rank].to_numpy(float) if max_rank in d.columns else np.maximum(d[rank].max(), 1)
    u = np.sort(d[rank].to_numpy(float) / np.maximum(scale, 1))
    ecdf = np.arange(1, len(u) + 1) / len(u)
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.plot(u, ecdf)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("SBC ECDF diagnostic")
    return fig


def plot_sbc_coverage_gg(x: Any, variables: Sequence[str] | None = None):
    d = _sbc_plot_data(x, variables)
    if "coverage" in d.columns:
        labels = d.get("parameter", d.get("variable", pd.Series(range(len(d))))).astype(str)
        values = d["coverage"].to_numpy(float)
    else:
        # Rank fractions within central 90% are a descriptive coverage proxy when
        # the Python SBC result records ranks but not interval coverage.
        max_rank = "draws" if "draws" in d.columns else "max_rank"
        scale = (
            d[max_rank].to_numpy(float) if max_rank in d.columns else np.maximum(d["rank"].max(), 1)
        )
        u = d["rank"].to_numpy(float) / np.maximum(scale, 1)
        labels = pd.Series(["central 90%"])
        values = np.array([np.mean((u >= 0.05) & (u <= 0.95))])
    plt = _mpl()
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_title("SBC empirical coverage")
    return fig


def plot_sbc_simulated_vs_estimated_gg(x: Any, variables: Sequence[str] | None = None):
    d = _sbc_plot_data(x, variables)
    pairs = next(
        (
            (a, b)
            for a, b in (("simulated", "estimated"), ("truth", "median"), ("value", "estimate"))
            if a in d.columns and b in d.columns
        ),
        None,
    )
    plt = _mpl()
    fig, ax = plt.subplots()
    if pairs is None:
        if "rank" not in d.columns:
            raise GP3BayesError("SBC simulated-versus-estimated statistics are unavailable.")
        xvals = np.arange(1, len(d) + 1)
        yvals = d["rank"].to_numpy(float)
        ax.scatter(xvals, yvals)
        ax.set_xlabel("SBC record")
        ax.set_ylabel("Rank")
    else:
        ax.scatter(d[pairs[0]], d[pairs[1]])
        lo = min(float(d[pairs[0]].min()), float(d[pairs[1]].min()))
        hi = max(float(d[pairs[0]].max()), float(d[pairs[1]].max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Simulated")
        ax.set_ylabel("Estimated")
    ax.set_title("SBC simulated versus estimated")
    return fig


__all__ = [
    "ModelEvidence",
    "ModelReport",
    "PriorSensitivity",
    "RecoveryResult",
    "SensitivityPlan",
    "SensitivitySuite",
    "SuiteError",
    "collect_model_evidence",
    "create_model_evidence_report",
    "create_sensitivity_suite_plan",
    "estimand_sensitivity_table",
    "group_deletion_sensitivity_table",
    "plot_estimand_sensitivity_gg",
    "plot_group_deletion_sensitivity",
    "plot_powerscale_sensitivity_gg",
    "plot_prior_sensitivity",
    "plot_prior_sensitivity_scenarios",
    "plot_random_slope_sensitivity",
    "plot_recovery_bias",
    "plot_recovery_coverage",
    "plot_recovery_estimates",
    "plot_recovery_fit_status",
    "plot_recovery_rmse",
    "plot_sbc_coverage_gg",
    "plot_sbc_ecdf_gg",
    "plot_sbc_rank_gg",
    "plot_sbc_simulated_vs_estimated_gg",
    "powerscale_sensitivity_table",
    "prior_sensitivity_scenario_table",
    "prior_sensitivity_table",
    "random_slope_sensitivity_table",
    "recovery_estimate_table",
    "recovery_fit_status_table",
    "recovery_parameter_table",
    "run_sensitivity_suite",
    "sbc_overview_table",
    "sbc_stats_table",
    "summarise_sensitivity_suite",
]
