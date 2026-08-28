"""Specification-closure, transformation replay, estimands and sensitivity.

This module adapts the frozen gp3bayes 0.5.0 specification-closure contracts
without weakening their governance: warnings never trigger silent exclusion,
reparameterisation or model selection, and invariance tolerances are explicit.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ModelContract, create_model_contract
from .exceptions import GP3BayesError
from .readiness import audit_model_readiness


def _closure_fixed_model_matrix(
    data: pd.DataFrame,
    contract: Any,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if getattr(contract, "family", None) == "binary":
        from .binary import _fixed_model_matrix as implementation
    elif getattr(contract, "family", None) == "duration":
        from .duration import _fixed_model_matrix as implementation
    else:
        raise GP3BayesError(
            "Fixed model-matrix replay is available only for binary or duration contracts."
        )
    return implementation(data, contract)


def _number(
    value: Any,
    name: str,
    lower: float = -math.inf,
    upper: float = math.inf,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if isinstance(value, bool):
        raise GP3BayesError(f"`{name}` must be one finite numeric value.")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(f"`{name}` must be one finite numeric value.") from exc
    if not math.isfinite(x):
        raise GP3BayesError(f"`{name}` must be one finite numeric value.")
    if (x <= lower if lower_open else x < lower) or (x >= upper if upper_open else x > upper):
        raise GP3BayesError(f"`{name}` is outside the approved range.")
    return x


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    x = _number(value, name)
    if x != math.floor(x) or x < minimum:
        raise GP3BayesError(f"`{name}` must be one integer >= {minimum}.")
    return int(x)


def _contract(contract: Any, family: str | None = None) -> ModelContract:
    if not isinstance(contract, ModelContract):
        raise GP3BayesError("`contract` must be a gp3bayes model contract.")
    if family is not None and contract.family != family:
        raise GP3BayesError(f"`contract` must use the `{family}` family.")
    return contract


def _mapping(contract: ModelContract, key: str) -> str | None:
    value = contract.mappings.get(key)
    return None if value is None else str(value)


def _check_row(
    check_id: str, domain: str, status: str, detail: str, n: float | int | None = None
) -> dict[str, Any]:
    return {"check_id": check_id, "domain": domain, "status": status, "detail": detail, "n": n}


def _worst_status(values: Sequence[str]) -> str:
    values = list(values)
    if "fail" in values:
        return "fail"
    if "warn" in values or "review" in values:
        return "review"
    if values and all(v == "not_applicable" for v in values):
        return "not_applicable"
    return "pass"


@dataclass(slots=True)
class ConditionBalance:
    status: str
    table: pd.DataFrame
    minimum_fraction: float
    warning_fraction: float
    failure_fraction: float
    interpretation: str = (
        "Condition balance is an observable design diagnostic. It does not by itself "
        "establish identifiability or adequacy."
    )


def summarise_condition_balance(
    data: pd.DataFrame,
    contract: ModelContract,
    warning_fraction: float = 0.10,
    failure_fraction: float = 0.02,
) -> ConditionBalance:
    contract = _contract(contract)
    warning = _number(warning_fraction, "warning_fraction", 0, 0.5, True, False)
    failure = _number(failure_fraction, "failure_fraction", 0, warning, False, True)
    column = _mapping(contract, "condition")
    if column is None:
        return ConditionBalance(
            "not_applicable",
            pd.DataFrame(),
            np.nan,
            warning,
            failure,
            "No focal condition is declared in the model contract.",
        )
    if column not in data:
        raise GP3BayesError(f"Condition column `{column}` is not present in `data`.")
    observed = data[column].dropna()
    if observed.empty:
        return ConditionBalance(
            "fail", pd.DataFrame(columns=["level", "n", "fraction"]), 0.0, warning, failure
        )
    counts = observed.value_counts(dropna=True, sort=False)
    table = pd.DataFrame({"level": counts.index.astype(str), "n": counts.to_numpy(int)})
    table["fraction"] = table["n"] / table["n"].sum()
    minimum = float(table["fraction"].min())
    status = (
        "fail"
        if len(table) != 2 or minimum < failure
        else ("review" if minimum < warning else "pass")
    )
    return ConditionBalance(status, table.reset_index(drop=True), minimum, warning, failure)


@dataclass(slots=True)
class BinaryGroupVariation:
    status: str
    group: str
    group_column: str | None
    table: pd.DataFrame
    n_no_variation: int
    fraction_no_variation: float = np.nan
    interpretation: str = (
        "Groups without observed binary variation are reported for design review. "
        "They are not automatically excluded."
    )


def summarise_binary_group_variation(
    data: pd.DataFrame, contract: ModelContract, group: str = "participant"
) -> BinaryGroupVariation:
    contract = _contract(contract, "binary")
    if group not in {"participant", "item"}:
        raise GP3BayesError("`group` must be either participant or item.")
    group_col = _mapping(contract, group)
    if group_col is None:
        return BinaryGroupVariation("not_applicable", group, None, pd.DataFrame(), 0)
    outcome = _mapping(contract, "outcome")
    if outcome is None or group_col not in data or outcome not in data:
        raise GP3BayesError(
            f"Binary group-variation audit requires columns: {group_col}, {outcome}."
        )
    rows = []
    for ident, frame in data.groupby(group_col, dropna=False, observed=False):
        values = pd.to_numeric(frame[outcome], errors="coerce").dropna()
        n0 = int((values == 0).sum())
        n1 = int((values == 1).sum())
        variation = values.nunique() > 1
        pattern = (
            "missing"
            if values.empty
            else ("all_zero" if n1 == 0 else ("all_one" if n0 == 0 else "variable"))
        )
        rows.append(
            {
                "group_id": str(ident),
                "n": len(values),
                "n_zero": n0,
                "n_one": n1,
                "variation": bool(variation),
                "pattern": pattern,
            }
        )
    table = pd.DataFrame(rows)
    n_no = int((~table["variation"]).sum()) if not table.empty else 0
    status = "fail" if table.empty or n_no == len(table) else ("review" if n_no else "pass")
    fraction = n_no / len(table) if len(table) else np.nan
    return BinaryGroupVariation(status, group, group_col, table, n_no, fraction)


@dataclass(slots=True)
class IdentifierPredictorAudit:
    status: str
    table: pd.DataFrame
    flagged: tuple[str, ...]
    interpretation: str = "Identifier-like predictor flags are conservative review signals only; explicit contract declarations are retained."


def identify_identifier_like_predictors(
    data: pd.DataFrame,
    contract: ModelContract,
    unique_fraction: float = 0.90,
    integer_fraction: float = 0.98,
    monotone_correlation: float = 0.98,
) -> IdentifierPredictorAudit:
    contract = _contract(contract)
    uf_thr = _number(unique_fraction, "unique_fraction", 0, 1)
    int_thr = _number(integer_fraction, "integer_fraction", 0, 1)
    cor_thr = _number(monotone_correlation, "monotone_correlation", 0, 1)
    predictors = tuple(contract.predictors)
    missing = [p for p in predictors if p not in data]
    if missing:
        raise GP3BayesError("Declared predictors are missing: " + ", ".join(missing) + ".")
    pattern = re.compile(r"(^|_)(id|index|row|participant|subject|item|stimulus|trial)(_|$)", re.I)
    rows = []
    for column in predictors:
        series = data[column]
        hint = bool(pattern.search(column))
        if not pd.api.types.is_numeric_dtype(series):
            rows.append(
                {
                    "predictor": column,
                    "numeric": False,
                    "unique_fraction": np.nan,
                    "integer_fraction": np.nan,
                    "row_order_correlation": np.nan,
                    "name_hint": hint,
                    "flagged": False,
                    "reason": "non_numeric",
                }
            )
            continue
        arr = pd.to_numeric(series, errors="coerce").to_numpy(float)
        finite = np.isfinite(arr)
        x = arr[finite]
        uf = len(np.unique(x)) / len(x) if len(x) else 0.0
        integer = float(np.mean(np.abs(x - np.round(x)) < 1e-8)) if len(x) else 0.0
        corr = np.nan
        if len(x) >= 3 and np.std(x, ddof=1) > 0:
            corr = abs(float(np.corrcoef(x, np.flatnonzero(finite) + 1)[0, 1]))
        flagged = (
            uf >= uf_thr
            and integer >= int_thr
            and (hint or (np.isfinite(corr) and corr >= cor_thr))
        )
        reasons = []
        if uf >= uf_thr:
            reasons.append("high_uniqueness")
        if integer >= int_thr:
            reasons.append("integer_like")
        if hint:
            reasons.append("identifier_name")
        if np.isfinite(corr) and corr >= cor_thr:
            reasons.append("row_order_like")
        rows.append(
            {
                "predictor": column,
                "numeric": True,
                "unique_fraction": uf,
                "integer_fraction": integer,
                "row_order_correlation": corr,
                "name_hint": hint,
                "flagged": bool(flagged),
                "reason": ";".join(reasons) if reasons else "none",
            }
        )
    table = pd.DataFrame(rows)
    flagged = (
        tuple(table.loc[table["flagged"].astype(bool), "predictor"].astype(str))
        if not table.empty
        else ()
    )  # noqa: E712
    return IdentifierPredictorAudit("review" if flagged else "pass", table, flagged)


@dataclass(slots=True)
class DurationExtremeReview:
    status: str
    table: pd.DataFrame
    n: int
    n_flagged: int
    mad_cutoff: float
    iqr_multiplier: float
    interpretation: str = "Extreme positive durations are retained and flagged for review; no observation is automatically removed."


def review_duration_extremes(
    data: pd.DataFrame,
    contract: ModelContract,
    mad_cutoff: float = 4,
    iqr_multiplier: float = 3,
) -> DurationExtremeReview:
    contract = _contract(contract, "duration")
    mad_cutoff = _number(mad_cutoff, "mad_cutoff", 0, math.inf, True)
    iqr_multiplier = _number(iqr_multiplier, "iqr_multiplier", 0, math.inf, True)
    outcome = _mapping(contract, "outcome")
    if outcome is None or outcome not in data:
        raise GP3BayesError("Duration outcome column is absent.")
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    if not np.isfinite(y).all() or np.any(y <= 0):
        raise GP3BayesError("Duration review requires finite strictly positive outcomes.")
    log_y = np.log(y)
    median = float(np.median(log_y))
    mad = float(np.median(np.abs(log_y - median)))
    robust_z = np.zeros_like(log_y) if mad == 0 else 0.6744897501960817 * (log_y - median) / mad
    q1, q3 = np.quantile(log_y, [0.25, 0.75], method="linear")
    iqr = float(q3 - q1)
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    flag_mad = np.abs(robust_z) > mad_cutoff
    flag_iqr = (log_y < lower) | (log_y > upper)
    flagged = flag_mad | flag_iqr
    table = pd.DataFrame(
        {
            "row": np.arange(1, len(y) + 1),
            "value": y,
            "log_value": log_y,
            "robust_z": robust_z,
            "mad_flag": flag_mad,
            "iqr_flag": flag_iqr,
            "flagged": flagged,
        }
    )
    n_flagged = int(flagged.sum())
    return DurationExtremeReview(
        "review" if n_flagged else "pass", table, len(y), n_flagged, mad_cutoff, iqr_multiplier
    )


@dataclass(slots=True)
class DurationBoundaryAudit:
    status: str
    checks: pd.DataFrame
    allowed_range: tuple[float, float] | None
    range_violations: tuple[int, ...]
    censor_col: str | None
    censored_rows: tuple[int, ...]
    candidate_columns: tuple[str, ...]
    interpretation: str = "The positive-duration contract is uncensored. Censoring signals or declared impossible values are contract failures, not automatic family switches."


def _censor_flags(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy(float)
        return x != 0
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "censored", "truncated", "right", "left"}).to_numpy(bool)


def audit_duration_boundaries(
    data: pd.DataFrame,
    contract: ModelContract,
    allowed_range: Sequence[float] | None = None,
    censor_col: str | None = None,
    detect_candidate_columns: bool = True,
) -> DurationBoundaryAudit:
    contract = _contract(contract, "duration")
    outcome = _mapping(contract, "outcome")
    if outcome is None or outcome not in data:
        raise GP3BayesError("Duration outcome column is absent.")
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
    checks = []
    positive = np.isfinite(y) & (y > 0)
    checks.append(
        _check_row(
            "strictly_positive",
            "duration_boundaries",
            "pass" if positive.all() else "fail",
            "All outcomes are finite and strictly positive."
            if positive.all()
            else "Some outcomes violate the positive-duration contract.",
            int((~positive).sum()),
        )
    )
    range_tuple: tuple[float, float] | None = None
    violations: tuple[int, ...] = ()
    if allowed_range is not None:
        vals = tuple(float(v) for v in allowed_range)
        if len(vals) != 2 or not np.isfinite(vals).all() or vals[0] <= 0 or vals[0] >= vals[1]:
            raise GP3BayesError(
                "`allowed_range` must contain two increasing positive finite values."
            )
        range_tuple = vals
        mask = (y < vals[0]) | (y > vals[1])
        violations = tuple((np.flatnonzero(mask) + 1).tolist())
        checks.append(
            _check_row(
                "declared_duration_range",
                "duration_boundaries",
                "fail" if violations else "pass",
                f"{len(violations)} rows fall outside the declared duration range.",
                len(violations),
            )
        )
    else:
        checks.append(
            _check_row(
                "declared_duration_range",
                "duration_boundaries",
                "not_applicable",
                "No allowed duration range was declared.",
            )
        )
    censored_rows: tuple[int, ...] = ()
    candidates: tuple[str, ...] = ()
    if censor_col is not None:
        if censor_col not in data:
            raise GP3BayesError(f"Censoring column `{censor_col}` is not present.")
        flags = _censor_flags(data[censor_col])
        censored_rows = tuple((np.flatnonzero(flags) + 1).tolist())
        checks.append(
            _check_row(
                "uncensored_contract",
                "duration_boundaries",
                "fail" if censored_rows else "pass",
                f"{len(censored_rows)} rows are marked censored or truncated."
                if censored_rows
                else "The supplied censoring indicator contains no censored observations.",
                len(censored_rows),
            )
        )
    elif detect_candidate_columns:
        candidates = tuple(
            c for c in data.columns if re.search(r"cens|censor|trunc|deadline", str(c), re.I)
        )
        checks.append(
            _check_row(
                "uncensored_contract",
                "duration_boundaries",
                "warn" if candidates else "pass",
                "Potential censoring/truncation columns require explicit review: "
                + ", ".join(candidates)
                if candidates
                else "No censoring-like column names were detected.",
                len(candidates),
            )
        )
    else:
        checks.append(
            _check_row(
                "uncensored_contract",
                "duration_boundaries",
                "not_applicable",
                "No censoring indicator was supplied or heuristically reviewed.",
            )
        )
    frame = pd.DataFrame(checks)
    return DurationBoundaryAudit(
        _worst_status(frame["status"].tolist()),
        frame,
        range_tuple,
        violations,
        censor_col,
        censored_rows,
        candidates,
    )


@dataclass(slots=True)
class StrictReadinessAudit:
    audit_version: str
    family: str
    ready: bool
    status: str
    status_counts: Mapping[str, int]
    checks: pd.DataFrame
    base_audit: Any
    condition_balance: ConditionBalance
    binary_group_variation: BinaryGroupVariation | None
    identifier_audit: IdentifierPredictorAudit
    rank: Mapping[str, Any]
    separation: Any
    duration_extremes: DurationExtremeReview | None
    duration_boundaries: DurationBoundaryAudit | None
    contract: ModelContract
    thresholds: Mapping[str, Any]
    interpretation: str = "Strict readiness is an observable-data gate. Passing does not establish convergence, posterior adequacy, predictive validity, or causal identification."


def audit_model_readiness_strict(
    data: pd.DataFrame,
    contract: ModelContract,
    condition_warning_fraction: float = 0.10,
    condition_failure_fraction: float = 0.02,
    identifier_unique_fraction: float = 0.90,
    duration_allowed_range: Sequence[float] | None = None,
    censor_col: str | None = None,
    run_separation: bool = True,
) -> StrictReadinessAudit:
    contract = _contract(contract)
    base = audit_model_readiness(data, contract)
    balance = summarise_condition_balance(
        data, contract, condition_warning_fraction, condition_failure_fraction
    )
    identifier = identify_identifier_like_predictors(data, contract, identifier_unique_fraction)
    rows = [
        _check_row(
            "overall_condition_balance",
            "design",
            "warn" if balance.status == "review" else balance.status,
            balance.interpretation
            if balance.status == "not_applicable"
            else f"Minimum observed condition fraction = {balance.minimum_fraction:.4g}.",
        ),
        _check_row(
            "identifier_like_predictors",
            "predictors",
            "warn" if identifier.status == "review" else identifier.status,
            "Identifier-like predictors require review: " + ", ".join(identifier.flagged)
            if identifier.flagged
            else "No declared predictor met the identifier-like heuristic.",
            len(identifier.flagged),
        ),
    ]
    # Reuse the design-support matrix builder for exact coding parity.
    try:
        matrix, _ = _closure_fixed_model_matrix(data, contract)
        rank = int(np.linalg.matrix_rank(matrix))
        columns = int(matrix.shape[1])
        rank_info = {"rank": rank, "columns": columns, "error": None}
        rows.append(
            _check_row(
                "fixed_effect_rank",
                "design",
                "fail" if rank < columns else "pass",
                f"Fixed-effects matrix rank {rank} of {columns}.",
            )
        )
    except Exception as exc:
        rank_info = {"rank": None, "columns": None, "error": str(exc)}  # type: ignore[dict-item]
        rows.append(
            _check_row(
                "fixed_effect_rank",
                "design",
                "fail",
                f"The fixed-effects matrix could not be constructed: {exc}",
            )
        )
    variation = None
    separation = None
    extremes = None
    boundaries = None
    if contract.family == "binary":
        variation = summarise_binary_group_variation(data, contract, "participant")
        rows.append(
            _check_row(
                "participant_binary_outcome_variation",
                "outcome",
                "warn" if variation.status == "review" else variation.status,
                f"{variation.n_no_variation} participant groups have no observed binary outcome variation.",
                variation.n_no_variation,
            )
        )
        if run_separation:
            try:
                from types import SimpleNamespace

                from .advanced_optional_workflows import detect_binary_separation

                separation = detect_binary_separation(
                    SimpleNamespace(contract=contract, prepared=SimpleNamespace(data=data))
                )
                detected = bool(
                    getattr(separation, "separation_detected", False)
                    if not isinstance(separation, Mapping)
                    else separation.get("separation_detected", False)
                )
                rows.append(
                    _check_row(
                        "fixed_effect_separation",
                        "design",
                        "warn" if detected else "pass",
                        "The fixed-effects logistic screen detected separation."
                        if detected
                        else "The fixed-effects logistic separation screen did not detect separation.",
                    )
                )
            except Exception as exc:
                rows.append(
                    _check_row(
                        "fixed_effect_separation",
                        "design",
                        "warn",
                        f"Separation screening could not be completed: {exc}",
                    )
                )
    else:
        extremes = review_duration_extremes(data, contract)
        rows.append(
            _check_row(
                "duration_extreme_review",
                "outcome",
                "warn" if extremes.status == "review" else extremes.status,
                f"{extremes.n_flagged} duration observations were flagged for extreme-value review.",
                extremes.n_flagged,
            )
        )
        boundaries = audit_duration_boundaries(data, contract, duration_allowed_range, censor_col)
        rows.extend(boundaries.checks.to_dict("records"))  # type: ignore[arg-type]
    base_checks = getattr(base, "checks", pd.DataFrame())
    if not isinstance(base_checks, pd.DataFrame):
        base_checks = pd.DataFrame()
    extras = pd.DataFrame(rows)
    # Keep the common readiness columns if available, otherwise retain closure rows.
    if not base_checks.empty:
        common = [c for c in base_checks.columns if c in extras.columns]
        combined = (
            pd.concat([base_checks[common], extras[common]], ignore_index=True)
            if common
            else extras
        )
    else:
        combined = extras
    statuses = combined["status"].astype(str)
    counts = {
        name: int((statuses == name).sum()) for name in ("pass", "warn", "fail", "not_applicable")
    }
    ready = counts["fail"] == 0
    status = "not_ready" if not ready else ("ready_with_warnings" if counts["warn"] else "ready")
    return StrictReadinessAudit(
        "0.2",
        contract.family,
        ready,
        status,
        counts,
        combined,
        base,
        balance,
        variation,
        identifier,
        rank_info,
        separation,
        extremes,
        boundaries,
        contract,
        {
            "condition_warning_fraction": condition_warning_fraction,
            "condition_failure_fraction": condition_failure_fraction,
            "identifier_unique_fraction": identifier_unique_fraction,
            "duration_allowed_range": duration_allowed_range,
        },
    )


@dataclass(slots=True)
class TransformationRecipe:
    recipe_version: str
    family: str
    contract: ModelContract
    transformations: Mapping[str, Any]
    fixed_formula: Any
    fixed_formula_text: str
    model_matrix_columns: tuple[str, ...]
    outcome_unit: str | None
    source_preparation_version: str
    interpretation: str = "This recipe replays only transformations recorded by gp3bayes. It does not infer new scaling, recode unseen levels, or repair missing data silently."


def create_transformation_recipe(prepared: Any) -> TransformationRecipe:
    if not hasattr(prepared, "contract") or not hasattr(prepared, "transformations"):
        raise GP3BayesError("`prepared` must be a gp3bayes binary or duration prepared object.")
    family = prepared.contract.family
    return TransformationRecipe(
        "0.2",
        family,
        prepared.contract,
        copy.deepcopy(prepared.transformations),
        prepared.fixed_formula,
        prepared.fixed_formula_text,
        tuple(prepared.model_matrix_columns),
        getattr(prepared, "outcome_unit", None),
        prepared.preparation_version,
    )


def _as_recipe(recipe: Any) -> TransformationRecipe:
    return (
        recipe if isinstance(recipe, TransformationRecipe) else create_transformation_recipe(recipe)
    )


def apply_transformation_recipe(
    new_data: pd.DataFrame,
    recipe: TransformationRecipe | Any,
    input_scale: str = "raw",
    require_outcome: bool = False,
    input_unit: str | None = None,
) -> pd.DataFrame:
    if not isinstance(new_data, pd.DataFrame):
        raise GP3BayesError("`new_data` must be a data frame.")
    recipe = _as_recipe(recipe)
    if input_scale not in {"raw", "prepared"}:
        raise GP3BayesError("`input_scale` must be raw or prepared.")
    data = new_data.copy(deep=True)
    outcome = _mapping(recipe.contract, "outcome")
    assert outcome is not None
    if require_outcome and outcome not in data:
        raise GP3BayesError(f"Outcome column `{outcome}` is required but absent.")
    if input_scale == "prepared":
        data.attrs["gp3bayes_transformation_recipe"] = recipe
        return data
    trans = recipe.transformations
    if recipe.family == "binary":
        if outcome in data:
            mapping = trans["outcome"]["mapping"]
            mapped = []
            for value in data[outcome]:
                if value in mapping:
                    mapped.append(mapping[value])
                elif str(value) in {"0", "1"}:
                    mapped.append(int(float(value)))
                else:
                    raise GP3BayesError(
                        "New binary outcome contains values absent from the recorded mapping."
                    )
            data[outcome] = np.asarray(mapped, dtype=int)
        condition = trans.get("condition")
        if condition is not None:
            column = condition.get("column") or _mapping(recipe.contract, "condition")
            if column not in data:
                raise GP3BayesError(f"Condition column `{column}` is absent.")
            coding = condition["coding"]
            values = []
            valid_codes = {float(v) for v in coding.values()}
            for value in data[column]:
                if value in coding:
                    values.append(float(coding[value]))
                else:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = np.nan
                    if numeric not in valid_codes:
                        raise GP3BayesError(
                            "New data contain condition values absent from the recorded coding."
                        )
                    values.append(numeric)
            data[column] = values
        for column, cfg in trans.get("numeric_scaling", {}).items():
            if column not in data:
                raise GP3BayesError(f"Scaled predictor `{column}` is absent.")
            values = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise GP3BayesError(f"Scaled predictor `{column}` must be finite and numeric.")
            data[column] = (values - float(cfg["center"])) / float(cfg["scale"])
    else:
        outcome_cfg = trans["outcome"]
        source_unit = outcome_cfg["source_unit"]
        if input_unit is not None and input_unit != source_unit:
            raise GP3BayesError(
                f"`input_unit` does not match the recipe source unit `{source_unit}`."
            )
        if outcome in data:
            y = pd.to_numeric(data[outcome], errors="coerce").to_numpy(float)
            if not np.isfinite(y).all() or np.any(y <= 0):
                raise GP3BayesError("New duration outcomes must be finite and strictly positive.")
            data[outcome] = y * float(outcome_cfg["multiplier"])
        condition = trans.get("condition")
        if condition is not None:
            column = _mapping(recipe.contract, "condition")
            assert column is not None
            if column not in data:
                raise GP3BayesError(f"Condition column `{column}` is absent.")
            coding = condition["coding"]
            valid_codes = {float(v) for v in coding.values()}
            values = []
            for value in data[column]:
                if value in coding:
                    values.append(float(coding[value]))
                else:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        numeric = np.nan
                    if numeric not in valid_codes:
                        raise GP3BayesError(
                            "New data contain condition values absent from the recorded coding."
                        )
                    values.append(numeric)
            data[column] = values
        for column, cfg in trans.get("scaled_columns", {}).items():
            if column not in data:
                raise GP3BayesError(f"Scaled predictor `{column}` is absent.")
            values = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise GP3BayesError(f"Scaled predictor `{column}` must be finite and numeric.")
            data[column] = (values - float(cfg["centre"])) / float(cfg["scale"])
    data.attrs["gp3bayes_transformation_recipe"] = recipe
    return data


def invert_transformation_recipe(
    data: pd.DataFrame, recipe: TransformationRecipe | Any
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    recipe = _as_recipe(recipe)
    out = data.copy(deep=True)
    trans = recipe.transformations
    outcome = _mapping(recipe.contract, "outcome")
    assert outcome is not None
    if recipe.family == "binary":
        for column, cfg in trans.get("numeric_scaling", {}).items():
            out[column] = pd.to_numeric(out[column]) * float(cfg["scale"]) + float(cfg["center"])
        condition = trans.get("condition")
        if condition is not None:
            column = condition.get("column") or _mapping(recipe.contract, "condition")
            inverse = {float(v): k for k, v in condition["coding"].items()}
            restored = [inverse.get(float(v)) for v in out[column]]
            if any(v is None for v in restored):
                raise GP3BayesError("Prepared condition values do not match the recorded coding.")
            out[column] = restored
        if outcome in out:
            mapping = trans["outcome"]["mapping"]
            inverse = {int(v): k for k, v in mapping.items()}
            restored = [inverse.get(int(v)) for v in out[outcome]]
            if any(v is None for v in restored):
                raise GP3BayesError("Prepared outcome values do not match the recorded mapping.")
            out[outcome] = restored
    else:
        for column, cfg in trans.get("scaled_columns", {}).items():
            out[column] = pd.to_numeric(out[column]) * float(cfg["scale"]) + float(cfg["centre"])
        condition = trans.get("condition")
        if condition is not None:
            column = _mapping(recipe.contract, "condition")
            assert column is not None
            inverse = {float(v): k for k, v in condition["coding"].items()}
            restored = [inverse.get(float(v)) for v in out[column]]
            if any(v is None for v in restored):
                raise GP3BayesError("Prepared condition values do not match the recorded coding.")
            out[column] = restored
        if outcome in out:
            multiplier = float(trans["outcome"]["multiplier"])
            out[outcome] = pd.to_numeric(out[outcome]) / multiplier
    return out


@dataclass(slots=True)
class TransformationReplayAudit:
    status: str
    table: pd.DataFrame
    tolerance: float
    replay_established: bool
    interpretation: str = "Replay checks only the transformations retained in the prepared analysis data; discarded rows cannot be recovered."


def validate_transformation_replay(
    prepared: Any, tolerance: float = 1e-10
) -> TransformationReplayAudit:
    tolerance = _number(tolerance, "tolerance", 0, math.inf)
    recipe = create_transformation_recipe(prepared)
    raw = invert_transformation_recipe(prepared.data, recipe)
    replay = apply_transformation_recipe(
        raw,
        recipe,
        require_outcome=True,
        input_unit=recipe.transformations.get("outcome", {}).get("source_unit")
        if recipe.family == "duration"
        else None,
    )
    rows = []
    for column in prepared.data.columns:
        a, b = prepared.data[column], replay[column]
        if pd.api.types.is_numeric_dtype(a):
            aa = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bb = pd.to_numeric(b, errors="coerce").to_numpy(float)
            error = float(np.nanmax(np.abs(aa - bb))) if len(aa) else 0.0
            ok = np.allclose(aa, bb, atol=tolerance, rtol=0, equal_nan=True)
        else:
            error = 0.0 if a.astype(str).equals(b.astype(str)) else np.inf
            ok = bool(np.array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy()))
        rows.append(
            {"column": column, "status": "pass" if ok else "fail", "maximum_absolute_error": error}
        )
    table = pd.DataFrame(rows)
    passed = bool((table["status"] == "pass").all())
    return TransformationReplayAudit("pass" if passed else "fail", table, tolerance, passed)


@dataclass(slots=True)
class Estimand:
    family: str
    primary_quantity: str
    draws: pd.DataFrame
    metadata: Mapping[str, Any]
    automatic_decision: bool = False
    interpretation: str = "The object contains posterior draws of a prespecified estimand. Credible intervals are not automatic causal or adequacy claims."


def _condition_metadata(prepared: Any) -> dict[str, Any]:
    trans = prepared.transformations.get("condition")
    if trans is None:
        raise GP3BayesError("A declared focal condition is required for this estimand.")
    coding = trans["coding"]
    source = tuple(trans.get("source_levels", tuple(coding)))
    if len(source) != 2:
        raise GP3BayesError("The focal condition must have exactly two recorded source levels.")
    return {
        "column": trans.get("column") or _mapping(prepared.contract, "condition"),
        "source_levels": source,
        "reference": float(coding[source[0]]),
        "focal": float(coding[source[1]]),
    }


def _target_data(fit: Any, target_data: pd.DataFrame | None, target_scale: str) -> pd.DataFrame:
    prepared = fit.specification.prepared
    if target_data is None:
        return prepared.data.copy()
    if target_scale not in {"prepared", "raw"}:
        raise GP3BayesError("`target_scale` must be prepared or raw.")
    return (
        target_data.copy()
        if target_scale == "prepared"
        else apply_transformation_recipe(target_data, prepared, require_outcome=False)
    )


def estimate_standardized_probability_contrast(
    fit: Any,
    target_data: pd.DataFrame | None = None,
    target_scale: str = "prepared",
    ndraws: int | None = None,
    include_group_effects: bool = False,
) -> Estimand:
    if getattr(fit, "family", None) != "binary" or getattr(fit, "fit_performed", False) is not True:
        raise GP3BayesError("`fit` must be an approved binary gp3bayes fit.")
    from .predictive import predict_model

    target = _target_data(fit, target_data, target_scale)
    condition = _condition_metadata(fit.specification.prepared)
    ref, foc = target.copy(), target.copy()
    ref[condition["column"]] = condition["reference"]
    foc[condition["column"]] = condition["focal"]
    p0 = predict_model(
        fit, ref, type="expected", include_group_effects=include_group_effects, ndraws=ndraws
    )
    p1 = predict_model(
        fit, foc, type="expected", include_group_effects=include_group_effects, ndraws=ndraws
    )
    a, b = np.asarray(p0.draws, float).mean(axis=1), np.asarray(p1.draws, float).mean(axis=1)
    eps = np.finfo(float).eps

    def odds(p):
        return p / np.maximum(1 - p, eps)

    draws = pd.DataFrame(
        {
            ".draw": np.arange(1, len(a) + 1),
            "reference_probability": a,
            "focal_probability": b,
            "probability_difference": b - a,
            "probability_ratio": b / np.maximum(a, eps),
            "odds_ratio_of_standardized_probabilities": odds(b) / np.maximum(odds(a), eps),
        }
    )
    return Estimand(
        "binary",
        "probability_difference",
        draws,
        {
            "condition_column": condition["column"],
            "reference_level": condition["source_levels"][0],
            "focal_level": condition["source_levels"][1],
            "target_rows": len(target),
            "include_group_effects": include_group_effects,
        },
    )


def estimate_standardized_duration_estimands(
    fit: Any,
    target_data: pd.DataFrame | None = None,
    target_scale: str = "prepared",
    predictive_quantile: float = 0.90,
    ndraws: int | None = None,
    include_group_effects: bool = False,
    seed: int = 1,
) -> Estimand:
    if (
        getattr(fit, "family", None) != "duration"
        or getattr(fit, "fit_performed", False) is not True
    ):
        raise GP3BayesError("`fit` must be an approved duration gp3bayes fit.")
    q = _number(predictive_quantile, "predictive_quantile", 0, 1, True, True)
    from .predictive import predict_model

    target = _target_data(fit, target_data, target_scale)
    condition = _condition_metadata(fit.specification.prepared)
    ref, foc = target.copy(), target.copy()
    ref[condition["column"]] = condition["reference"]
    foc[condition["column"]] = condition["focal"]
    lin0 = predict_model(
        fit, ref, type="linear", include_group_effects=include_group_effects, ndraws=ndraws
    )
    lin1 = predict_model(
        fit, foc, type="linear", include_group_effects=include_group_effects, ndraws=ndraws
    )
    pr0 = predict_model(
        fit,
        ref,
        type="predictive",
        include_group_effects=include_group_effects,
        ndraws=ndraws,
        seed=seed,
    )
    pr1 = predict_model(
        fit,
        foc,
        type="predictive",
        include_group_effects=include_group_effects,
        ndraws=ndraws,
        seed=seed,
    )
    l0, l1 = np.asarray(lin0.draws, float), np.asarray(lin1.draws, float)
    p0, p1 = np.asarray(pr0.draws, float), np.asarray(pr1.draws, float)
    m0, m1 = np.exp(l0).mean(axis=1), np.exp(l1).mean(axis=1)
    q0 = np.quantile(p0, q, axis=1, method="median_unbiased")
    q1 = np.quantile(p1, q, axis=1, method="median_unbiased")
    eps = np.finfo(float).eps
    draws = pd.DataFrame(
        {
            ".draw": np.arange(1, len(m0) + 1),
            "average_log_duration_contrast": (l1 - l0).mean(axis=1),
            "reference_average_conditional_median": m0,
            "focal_average_conditional_median": m1,
            "conditional_median_difference": m1 - m0,
            "conditional_median_ratio": m1 / np.maximum(m0, eps),
            "reference_predictive_quantile": q0,
            "focal_predictive_quantile": q1,
            "predictive_quantile_difference": q1 - q0,
            "predictive_quantile_ratio": q1 / np.maximum(q0, eps),
        }
    )
    return Estimand(
        "duration",
        "conditional_median_ratio",
        draws,
        {
            "condition_column": condition["column"],
            "reference_level": condition["source_levels"][0],
            "focal_level": condition["source_levels"][1],
            "target_rows": len(target),
            "predictive_quantile": q,
            "outcome_unit": getattr(fit, "outcome_unit", None),
            "include_group_effects": include_group_effects,
            "seed": seed,
        },
    )


def _summary_vector(values: Sequence[float], probs: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, float)
    if arr.size < 2 or not np.isfinite(arr).all():
        raise GP3BayesError(
            "Estimand draws must be a finite numeric vector with at least two values."
        )
    p = tuple(float(v) for v in probs)
    q = np.quantile(arr, p, method="median_unbiased")
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)),
        "median": float(np.median(arr)),
        "lower": float(q[0]),
        "middle": float(q[1]),
        "upper": float(q[2]),
    }


def summarise_estimand_draws(
    x: Estimand | Sequence[float],
    quantities: Sequence[str] | str | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    p = tuple(float(v) for v in probs)
    if len(p) != 3 or not 0 <= p[0] < p[1] < p[2] <= 1:
        raise GP3BayesError("`probs` must contain three strictly increasing probabilities.")
    if isinstance(x, Estimand):
        available = [c for c in x.draws.columns if c != ".draw"]
        selected = (
            available
            if quantities is None
            else ([quantities] if isinstance(quantities, str) else list(quantities))
        )
        missing = [q for q in selected if q not in available]
        if missing:
            raise GP3BayesError("Unknown estimand quantities: " + ", ".join(missing) + ".")
        return pd.DataFrame([{"quantity": q, **_summary_vector(x.draws[q], p)} for q in selected])  # type: ignore[arg-type]
    return pd.DataFrame([_summary_vector(x, p)])


@dataclass(slots=True)
class RandomSlopeSensitivityPlan:
    plan_version: str
    family: str
    intercept_only: Mapping[str, Any]
    random_slope: Mapping[str, Any]
    automatic_selection: bool = False
    interpretation: str = "The plan exposes both approved structures for sensitivity analysis; no structure is selected automatically."


def _clone_contract(
    contract: ModelContract, *, random_slope: bool | None = None, outcome_unit: str | None = None
) -> ModelContract:
    return create_model_contract(
        family=contract.family,
        outcome_col=_mapping(contract, "outcome"),  # type: ignore[arg-type]
        participant_col=_mapping(contract, "participant"),  # type: ignore[arg-type]
        item_col=_mapping(contract, "item"),
        trial_col=_mapping(contract, "trial"),
        condition_col=_mapping(contract, "condition"),
        time_col=_mapping(contract, "time"),
        predictors=contract.predictors,
        interaction=contract.interaction,
        random_slope=contract.random_slope if random_slope is None else random_slope,
        outcome_unit=contract.outcome_unit if outcome_unit is None else outcome_unit,
        notes=contract.notes,
    )


def _prior_cfg(spec: Any) -> dict[str, Any]:
    table = spec.priors.table

    def row(cls: str):
        z = table.loc[table["parameter_class"] == cls]
        return None if len(z) != 1 else z.iloc[0]

    intercept, b, sd, sigma, cor = row("Intercept"), row("b"), row("sd"), row("sigma"), row("cor")
    advanced = getattr(spec, "advanced_priors", None)
    return {
        "baseline": spec.priors.baseline,
        "intercept_scale": float(intercept["scale"]),
        "coefficient_scale": float(b["scale"]),
        "group_sd_scale": float(sd["scale"]),
        "residual_scale": None if sigma is None else float(sigma["scale"]),
        "correlation_eta": 2.0 if cor is None else float(cor["shape"]),
        "student_df": float(sd["df"]),
        "advanced": advanced is not None,
        "main_effect_scale": float(
            getattr(advanced, "get", lambda *_: b["scale"])("main_effect_scale", b["scale"])
        )
        if advanced is not None
        else float(b["scale"]),
        "interaction_scale": advanced.get("interaction_scale")
        if isinstance(advanced, Mapping)
        else None,
    }


def _reprepare(
    spec: Any,
    contract: ModelContract,
    data: pd.DataFrame | None = None,
    *,
    condition_coding: Sequence[float] | None = None,
) -> Any:
    prepared = spec.prepared
    recipe = create_transformation_recipe(prepared)
    raw = (
        invert_transformation_recipe(prepared.data if data is None else data, recipe)
        if data is None
        else data
    )
    source_levels = (
        tuple(prepared.transformations.get("condition", {}).get("source_levels", ())) or None
    )
    if spec.family == "binary":
        from .binary import prepare_hierarchical_binary_data

        scaling = tuple(prepared.transformations.get("numeric_scaling", {}))
        return prepare_hierarchical_binary_data(
            raw,
            contract,
            outcome_mapping=prepared.transformations["outcome"]["mapping"],
            condition_levels=source_levels,
            condition_coding=condition_coding
            or tuple(prepared.transformations.get("condition", {}).get("coding", {}).values())
            or (-0.5, 0.5),
            scale_predictors=[p for p in scaling if p in contract.predictors],
            scale_time=bool(_mapping(contract, "time") in scaling),
            missing="error",
        )
    from .duration import prepare_hierarchical_duration_data

    scaling = tuple(prepared.transformations.get("scaled_columns", {}))
    outcfg = prepared.transformations["outcome"]
    return prepare_hierarchical_duration_data(
        raw,
        contract,
        condition_levels=source_levels,
        condition_coding=condition_coding
        or tuple(prepared.transformations.get("condition", {}).get("coding", {}).values())
        or (-0.5, 0.5),
        scale_predictors=[p for p in scaling if p in contract.predictors],
        scale_time=bool(_mapping(contract, "time") in scaling),
        outcome_multiplier=float(outcfg["multiplier"]),
        converted_unit=outcfg["analysis_unit"]
        if outcfg["analysis_unit"] != outcfg["source_unit"]
        else None,
        missing="error",
    )


def _rebuild(
    prepared: Any,
    template: Any,
    *,
    baseline: float | None = None,
    coefficient_scale: float | None = None,
    interaction_scale: float | None = None,
) -> Any:
    cfg = _prior_cfg(template)
    baseline = cfg["baseline"] if baseline is None else baseline
    coefficient_scale = cfg["coefficient_scale"] if coefficient_scale is None else coefficient_scale
    advanced = (
        hasattr(template, "advanced_priors")
        and getattr(template, "advanced_priors", None) is not None
    )
    if template.family == "binary":
        if advanced and template.contract.interaction:
            from .advanced_optional_workflows import specify_binary_model_with_interaction_prior

            return specify_binary_model_with_interaction_prior(
                prepared,
                baseline,
                cfg["intercept_scale"],
                coefficient_scale,
                cfg["interaction_scale"] if interaction_scale is None else interaction_scale,
                cfg["group_sd_scale"],
                cfg["correlation_eta"],
                cfg["student_df"],
            )
        from .binary import specify_binary_model

        return specify_binary_model(
            prepared,
            baseline,
            cfg["intercept_scale"],
            coefficient_scale,
            cfg["group_sd_scale"],
            cfg["correlation_eta"],
            cfg["student_df"],
        )
    if advanced and template.contract.interaction:
        from .advanced_optional_workflows import specify_duration_model_with_interaction_prior

        return specify_duration_model_with_interaction_prior(
            prepared,
            baseline,
            cfg["intercept_scale"],
            coefficient_scale,
            cfg["interaction_scale"] if interaction_scale is None else interaction_scale,
            cfg["group_sd_scale"],
            cfg["residual_scale"],
            cfg["correlation_eta"],
            cfg["student_df"],
        )
    from .duration import specify_duration_model

    return specify_duration_model(
        prepared,
        baseline,
        cfg["intercept_scale"],
        coefficient_scale,
        cfg["group_sd_scale"],
        cfg["residual_scale"],
        cfg["correlation_eta"],
        cfg["student_df"],
    )


def create_random_slope_sensitivity_plan(specification: Any) -> RandomSlopeSensitivityPlan:
    if (
        not hasattr(specification, "prepared")
        or _mapping(specification.contract, "condition") is None
    ):
        raise GP3BayesError(
            "Random-slope sensitivity requires an approved model specification with a focal condition."
        )

    def make(flag: bool):
        contract = _clone_contract(specification.contract, random_slope=flag)
        try:
            prepared = _reprepare(specification, contract)
            spec = _rebuild(prepared, specification)
            return {
                "ready": bool(prepared.audit.ready),
                "contract": contract,
                "prepared": prepared,
                "specification": spec if prepared.audit.ready else None,
            }
        except Exception as exc:
            return {
                "ready": False,
                "contract": contract,
                "prepared": None,
                "specification": None,
                "error": str(exc),
            }

    return RandomSlopeSensitivityPlan("0.2", specification.family, make(False), make(True))


@dataclass(slots=True)
class GroupDeletionSensitivityPlan:
    plan_version: str
    family: str
    group: str
    group_column: str
    units: tuple[str, ...]
    table: pd.DataFrame
    specification: Any
    max_units: int
    automatic_exclusion: bool = False
    interpretation: str = "Omission fits are sensitivity analyses only; no participant or item is automatically excluded because an estimate changes."


def create_group_deletion_sensitivity_plan(
    specification: Any,
    group: str = "participant",
    units: Sequence[str] | None = None,
    max_units: int = 20,
) -> GroupDeletionSensitivityPlan:
    if group not in {"participant", "item"}:
        raise GP3BayesError("`group` must be participant or item.")
    max_units = _integer(max_units, "max_units", 1)
    column = _mapping(specification.contract, group)
    if column is None:
        raise GP3BayesError("The requested grouping variable is not declared.")
    available = tuple(pd.unique(specification.prepared.data[column].astype(str)))
    if units is None:
        if len(available) > max_units:
            raise GP3BayesError(
                f"The design contains {len(available)} {group} levels. Supply `units` explicitly to avoid an unbounded refitting request."
            )
        selected = available
    else:
        selected = tuple(str(v) for v in units)
    if len(set(selected)) != len(selected) or any(not v for v in selected):
        raise GP3BayesError("`units` must contain unique non-empty group identifiers.")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise GP3BayesError("Unknown omission units: " + ", ".join(unknown) + ".")
    rows = []
    for unit in selected:
        subset = specification.prepared.data[
            specification.prepared.data[column].astype(str) != unit
        ]
        try:
            audit = audit_model_readiness(subset, specification.contract)
            ready = bool(audit.ready)
            status = audit.status
        except Exception:
            ready = False
            status = "error"
        rows.append(
            {"omitted_unit": unit, "n_remaining": len(subset), "ready": ready, "status": status}
        )
    return GroupDeletionSensitivityPlan(
        "0.2",
        specification.family,
        group,
        column,
        selected,
        pd.DataFrame(rows),
        specification,
        max_units,
    )


def create_contrast_coding_sensitivity_specification(
    specification: Any, condition_coding: Sequence[float], baseline: float
) -> Any:
    coding = tuple(float(v) for v in condition_coding)
    if len(coding) != 2 or not np.isfinite(coding).all() or coding[0] == coding[1]:
        raise GP3BayesError("`condition_coding` must contain two distinct finite numeric values.")
    raw = invert_transformation_recipe(specification.prepared.data, specification.prepared)
    if specification.family == "binary":
        from .binary import prepare_hierarchical_binary_data

        t = specification.prepared.transformations
        levels = t["condition"]["source_levels"]
        scales = tuple(t.get("numeric_scaling", {}))
        prepared = prepare_hierarchical_binary_data(
            raw,
            specification.contract,
            outcome_mapping=t["outcome"]["mapping"],
            condition_levels=levels,
            condition_coding=coding,
            scale_predictors=[p for p in scales if p in specification.contract.predictors],
            scale_time=bool(_mapping(specification.contract, "time") in scales),
            missing="error",
        )
    else:
        from .duration import prepare_hierarchical_duration_data

        t = specification.prepared.transformations
        levels = t["condition"]["source_levels"]
        scales = tuple(t.get("scaled_columns", {}))
        out = t["outcome"]
        prepared = prepare_hierarchical_duration_data(  # type: ignore[assignment]
            raw,
            specification.contract,
            condition_levels=levels,
            condition_coding=coding,
            scale_predictors=[p for p in scales if p in specification.contract.predictors],
            scale_time=bool(_mapping(specification.contract, "time") in scales),
            outcome_multiplier=out["multiplier"],
            converted_unit=out["analysis_unit"]
            if out["analysis_unit"] != out["source_unit"]
            else None,
            missing="error",
        )
    return _rebuild(prepared, specification, baseline=baseline)


def create_predictor_scaling_sensitivity_specification(
    specification: Any,
    predictor: str,
    scale_factor: float,
    coefficient_scale: float,
    interaction_scale: float | None = None,
) -> Any:
    scale_factor = _number(scale_factor, "scale_factor", 0, math.inf, True)
    coefficient_scale = _number(coefficient_scale, "coefficient_scale", 0, math.inf, True)
    if predictor not in specification.contract.predictors:
        raise GP3BayesError("`predictor` must be declared in the approved contract.")
    prepared = copy.deepcopy(specification.prepared)
    if specification.family == "binary":
        registry = prepared.transformations["numeric_scaling"]
        if predictor not in registry:
            raise GP3BayesError("The requested predictor was not scaled during binary preparation.")
        prepared.data[predictor] = prepared.data[predictor] / scale_factor
        registry[predictor]["scale"] *= scale_factor
    else:
        registry = prepared.transformations["scaled_columns"]
        if predictor not in registry:
            raise GP3BayesError(
                "The requested predictor was not scaled during duration preparation."
            )
        prepared.data[predictor] = prepared.data[predictor] / scale_factor
        registry[predictor]["scale"] *= scale_factor
    return _rebuild(
        prepared,
        specification,
        coefficient_scale=coefficient_scale,
        interaction_scale=interaction_scale,
    )


def create_duration_unit_sensitivity_specification(
    specification: Any, multiplier: float, new_unit: str
) -> Any:
    if specification.family != "duration":
        raise GP3BayesError("`specification` must be an approved duration specification.")
    multiplier = _number(multiplier, "multiplier", 0, math.inf, True)
    if not isinstance(new_unit, str) or not new_unit.strip():
        raise GP3BayesError("`new_unit` must be non-empty.")
    prepared = copy.deepcopy(specification.prepared)
    outcome = _mapping(specification.contract, "outcome")
    assert outcome
    prepared.data[outcome] = prepared.data[outcome] * multiplier
    contract = _clone_contract(specification.contract, outcome_unit=new_unit)
    object.__setattr__(prepared, "contract", contract)
    object.__setattr__(prepared, "outcome_unit", new_unit)
    object.__setattr__(prepared, "audit", audit_model_readiness(prepared.data, contract))
    prepared.transformations["outcome"]["analysis_unit"] = new_unit
    prepared.transformations["outcome"]["multiplier"] *= multiplier
    cfg = _prior_cfg(specification)
    return _rebuild(prepared, specification, baseline=cfg["baseline"] * multiplier)


@dataclass(slots=True)
class EstimandSensitivity:
    status: str
    family: str
    quantity: str
    reference: Estimand
    alternatives: Mapping[str, Estimand]
    table: pd.DataFrame
    robustness_established: bool = False
    interpretation: str = "Sensitivity is reported as posterior-summary shifts; no universal threshold for robustness is imposed."


def compare_estimand_sensitivity(
    reference: Estimand, alternatives: Mapping[str, Estimand], quantity: str | None = None
) -> EstimandSensitivity:
    if not isinstance(reference, Estimand):
        raise GP3BayesError("`reference` must be a gp3bayes estimand.")
    if not alternatives:
        raise GP3BayesError("`alternatives` must be a non-empty named list of estimands.")
    q = reference.primary_quantity if quantity is None else quantity
    if q not in reference.draws:
        raise GP3BayesError("Unknown reference quantity.")
    ref = _summary_vector(reference.draws[q], (0.025, 0.5, 0.975))  # type: ignore[arg-type]
    rows = []
    for name, x in alternatives.items():
        if not isinstance(x, Estimand) or q not in x.draws:
            raise GP3BayesError("Every alternative must contain the requested estimand quantity.")
        s = _summary_vector(x.draws[q], (0.025, 0.5, 0.975))  # type: ignore[arg-type]
        pooled = float(np.std(np.r_[reference.draws[q], x.draws[q]], ddof=1))
        rows.append(
            {
                "alternative": name,
                "reference_median": ref["median"],
                "alternative_median": s["median"],
                "median_shift": s["median"] - ref["median"],
                "standardized_shift": abs(s["median"] - ref["median"]) / pooled
                if pooled > 0
                else np.nan,
                "reference_lower": ref["lower"],
                "reference_upper": ref["upper"],
                "alternative_lower": s["lower"],
                "alternative_upper": s["upper"],
            }
        )
    return EstimandSensitivity(
        "review", reference.family, q, reference, alternatives, pd.DataFrame(rows)
    )


@dataclass(slots=True)
class EstimandInvarianceAudit:
    status: str
    quantity: str
    absolute_median_shift: float
    tolerance: float
    comparison: EstimandSensitivity
    invariance_established: bool
    interpretation: str = "The tolerance is user-declared and quantity-specific; a pass documents numerical invariance under that tolerance only."


def audit_estimand_invariance(
    reference: Estimand, alternative: Estimand, quantity: str | None = None, tolerance: float = 0.0
) -> EstimandInvarianceAudit:
    tolerance = _number(tolerance, "tolerance", 0, math.inf)
    q = reference.primary_quantity if quantity is None else quantity
    comparison = compare_estimand_sensitivity(reference, {"alternative": alternative}, q)
    shift = abs(float(comparison.table.iloc[0]["median_shift"]))
    ok = shift <= tolerance
    return EstimandInvarianceAudit("pass" if ok else "review", q, shift, tolerance, comparison, ok)


@dataclass(slots=True)
class DurationUnitInvarianceAudit:
    status: str
    ratio_shift: Mapping[str, float]
    relative_scaling_error: Mapping[str, float]
    tolerance: float
    multiplier: float
    invariance_established: bool
    interpretation: str = "Unit-free ratios should remain stable under pure unit conversion, while absolute duration quantities should scale by the declared multiplier."


def audit_duration_unit_invariance(
    reference: Estimand, converted: Estimand, multiplier: float, tolerance: float = 0.02
) -> DurationUnitInvarianceAudit:
    if reference.family != "duration" or converted.family != "duration":
        raise GP3BayesError("Both estimands must be duration gp3bayes estimands.")
    multiplier = _number(multiplier, "multiplier", 0, math.inf, True)
    tolerance = _number(tolerance, "tolerance", 0, math.inf)
    ratios = {
        name: abs(float(np.median(reference.draws[name])) - float(np.median(converted.draws[name])))
        for name in ("conditional_median_ratio", "predictive_quantile_ratio")
    }
    absolutes = {}
    for name in (
        "reference_average_conditional_median",
        "focal_average_conditional_median",
        "reference_predictive_quantile",
        "focal_predictive_quantile",
    ):
        expected = float(np.median(reference.draws[name])) * multiplier
        observed = float(np.median(converted.draws[name]))
        absolutes[name] = abs(observed - expected) / max(abs(expected), np.finfo(float).eps)
    ok = all(v <= tolerance for v in ratios.values()) and all(
        v <= tolerance for v in absolutes.values()
    )
    return DurationUnitInvarianceAudit(
        "pass" if ok else "review", ratios, absolutes, tolerance, multiplier, ok
    )


@dataclass(slots=True)
class SensitivityRun:
    status: str
    plan: Any
    backend: str
    reference_estimand: Estimand | None
    results: Mapping[str, Any]
    summary: pd.DataFrame
    automatic_exclusion: bool = False
    automatic_selection: bool = False
    reference_fit: Any | None = None
    interpretation: str = "Sensitivity analysis reports declared perturbations without automatic exclusion or model selection."


def _fit_spec(
    spec: Any,
    backend: str,
    chains: int,
    iter: int,
    warmup: int,
    cores: int,
    seed: int,
    adapt_delta: float,
    max_treedepth: int,
    refresh: int,
):
    from .advanced_optional_workflows import fit_binary_model_backend, fit_duration_model_backend

    fn = fit_binary_model_backend if spec.family == "binary" else fit_duration_model_backend
    return fn(
        spec,
        backend=backend,
        chains=chains,
        iter=iter,
        warmup=warmup,
        cores=cores,
        seed=seed,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        refresh=refresh,
    )


def _primary_estimand(fit: Any, ndraws: int | None, seed: int) -> Estimand:
    return (
        estimate_standardized_probability_contrast(fit, ndraws=ndraws)
        if fit.family == "binary"
        else estimate_standardized_duration_estimands(fit, ndraws=ndraws, seed=seed)
    )


def run_group_deletion_sensitivity(
    plan: GroupDeletionSensitivityPlan,
    backend: str = "pymc",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 1,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
    ndraws: int | None = None,
    retain_fits: bool = False,
) -> SensitivityRun:
    if not isinstance(plan, GroupDeletionSensitivityPlan):
        raise GP3BayesError("`plan` must be a group-deletion sensitivity plan.")
    reference_fit = _fit_spec(
        plan.specification,
        backend,
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )
    ref = _primary_estimand(reference_fit, ndraws, seed)
    rs = summarise_estimand_draws(ref, ref.primary_quantity).iloc[0]
    results = {}
    fits = {}
    rows = []
    for i, unit in enumerate(plan.units, 1):
        try:
            data = plan.specification.prepared.data[
                plan.specification.prepared.data[plan.group_column].astype(str) != unit
            ]
            raw = invert_transformation_recipe(data, plan.specification.prepared)
            prepared = _reprepare(plan.specification, plan.specification.contract, raw)
            spec = _rebuild(prepared, plan.specification)
            fit = _fit_spec(
                spec,
                backend,
                chains,
                iter,
                warmup,
                cores,
                seed + i,
                adapt_delta,
                max_treedepth,
                refresh,
            )
            est = _primary_estimand(fit, ndraws, seed + i)
            s = summarise_estimand_draws(est, est.primary_quantity).iloc[0]
            results[unit] = est
            fits[unit] = fit
            rows.append(
                {
                    "omitted_unit": unit,
                    "status": "completed",
                    "median": s["median"],
                    "lower": s["lower"],
                    "upper": s["upper"],
                    "median_shift": s["median"] - rs["median"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "omitted_unit": unit,
                    "status": "error",
                    "median": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "median_shift": np.nan,
                    "error": str(exc),
                }
            )
    out = SensitivityRun(
        "review",
        plan,
        backend,
        ref,
        results,
        pd.DataFrame(rows),
        reference_fit=reference_fit if retain_fits else None,
    )
    out.reference_fit = reference_fit if retain_fits else None
    out.fits = fits if retain_fits else None  # type: ignore[attr-defined]
    return out


def run_random_slope_sensitivity(
    plan: RandomSlopeSensitivityPlan,
    backend: str = "pymc",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 1,
    seed: int = 1,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
    ndraws: int | None = None,
    retain_fits: bool = False,
) -> Any:
    if not isinstance(plan, RandomSlopeSensitivityPlan):
        raise GP3BayesError("`plan` must be a random-slope sensitivity plan.")
    if not plan.intercept_only["ready"] or not plan.random_slope["ready"]:
        raise GP3BayesError("Both structural specifications must pass readiness before refitting.")
    fits = {}
    est = {}
    for i, (name, node) in enumerate(
        (("random_intercept", plan.intercept_only), ("random_slope", plan.random_slope))
    ):
        fits[name] = _fit_spec(
            node["specification"],
            backend,
            chains,
            iter,
            warmup,
            cores,
            seed + i,
            adapt_delta,
            max_treedepth,
            refresh,
        )
        est[name] = _primary_estimand(fits[name], ndraws, seed + i)
    comparison = compare_estimand_sensitivity(
        est["random_intercept"], {"random_slope": est["random_slope"]}
    )
    return {
        "status": "review",
        "plan": plan,
        "backend": backend,
        "estimands": est,
        "comparison": comparison,
        "fits": fits if retain_fits else None,
        "automatic_selection": False,
        "interpretation": "No random-effects structure is selected automatically.",
    }


def check_binary_ppc_details(
    fit: Any, draws: int = 300, seed: int = 1, calibration_bins: int = 10, sparse_cell_min: int = 3
) -> Mapping[str, Any]:
    if getattr(fit, "family", None) != "binary":
        raise GP3BayesError("`fit` must be an approved binary gp3bayes fit.")
    from .predictive import binary_calibration_table, predict_model

    pred = predict_model(fit, type="predictive", ndraws=draws, seed=seed)
    expected = predict_model(fit, type="expected", ndraws=draws)
    calib = binary_calibration_table(expected, bins=calibration_bins)
    y = np.asarray(pred.observed, float)
    yrep = np.asarray(pred.draws, float)
    prepared = fit.specification.prepared
    groups = {}
    for key in ("participant", "item"):
        col = _mapping(prepared.contract, key)
        if col:
            rows = []
            for level, idx in prepared.data.groupby(col, observed=False).groups.items():
                ids = np.asarray(list(idx), int)
                obs = float(y[ids].mean())
                rep = yrep[:, ids].mean(axis=1)
                q = np.quantile(rep, [0.025, 0.5, 0.975], method="median_unbiased")
                rows.append(
                    {
                        key: str(level),
                        "n": len(ids),
                        "observed_rate": obs,
                        "replicated_mean": float(rep.mean()),
                        "lower": q[0],
                        "median": q[1],
                        "upper": q[2],
                        "sparse": len(ids) < sparse_cell_min,
                    }
                )
            groups[key] = pd.DataFrame(rows)
    return {
        "family": "binary",
        "draws": draws,
        "seed": seed,
        "calibration": calib,
        "groups": groups,
        "automatic_exclusion": False,
        "adequacy_established": False,
    }


def check_duration_ppc_details(
    fit: Any,
    draws: int = 300,
    seed: int = 1,
    quantiles: Sequence[float] = (0.5, 0.9, 0.95),
    tail_threshold: float | None = None,
) -> Mapping[str, Any]:
    if getattr(fit, "family", None) != "duration":
        raise GP3BayesError("`fit` must be an approved duration gp3bayes fit.")
    from .predictive import predict_model

    pred = predict_model(fit, type="predictive", ndraws=draws, seed=seed)
    y = np.asarray(pred.observed, float)
    yrep = np.asarray(pred.draws, float)
    qrows = []
    for q in quantiles:
        observed = float(np.quantile(y, q, method="median_unbiased"))
        rep = np.quantile(yrep, q, axis=1, method="median_unbiased")
        interval = np.quantile(rep, [0.025, 0.5, 0.975], method="median_unbiased")
        qrows.append(
            {
                "quantile": q,
                "observed": observed,
                "replicated_mean": float(rep.mean()),
                "lower": interval[0],
                "median": interval[1],
                "upper": interval[2],
            }
        )
    if tail_threshold is None:
        tail_threshold = float(np.quantile(y, 0.95, method="median_unbiased"))
    tail_obs = float(np.mean(y > tail_threshold))
    tail_rep = np.mean(yrep > tail_threshold, axis=1)
    return {
        "family": "duration",
        "draws": draws,
        "seed": seed,
        "quantiles": pd.DataFrame(qrows),
        "tail_threshold": tail_threshold,
        "observed_tail_rate": tail_obs,
        "replicated_tail_rate": tail_rep,
        "automatic_exclusion": False,
        "adequacy_established": False,
    }


@dataclass(slots=True)
class KFoldCV:
    family: str
    K: int
    folds: str
    group: str | None
    joint: str
    table: pd.DataFrame
    total_elpd: float
    se_elpd: float
    fits: Any = None
    automatic_selection: bool = False
    interpretation: str = (
        "K-fold predictive summaries support comparison but do not select a model automatically."
    )


def compute_kfold_cv(
    fit: Any,
    K: int = 10,
    folds: str = "random",
    group: str | None = None,
    joint: str = "obs",
    save_fits: bool = False,
    seed: int = 1,
) -> KFoldCV:
    if getattr(fit, "fit_performed", False) is not True or getattr(fit, "family", None) not in {
        "binary",
        "duration",
    }:
        raise GP3BayesError("`fit` must be a gp3bayes_fit.")
    K = _integer(K, "K", 2)
    if folds not in {"random", "stratified", "grouped"}:
        raise GP3BayesError("`folds` must be random, stratified, or grouped.")
    # The restricted Python adaptation uses pointwise PSIS log predictive density as a
    # deterministic K-fold surrogate when stored log-likelihood draws are available.
    # This avoids hidden refits; true K-fold refitting remains explicit via backend workflows.
    from .advanced_optional_workflows import compute_psis_loo

    loo = compute_psis_loo(fit)
    pt = loo.pointwise.copy()
    n = len(pt)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    fold = np.empty(n, int)
    fold[order] = np.arange(n) % K + 1
    if folds == "grouped":
        if not group:
            raise GP3BayesError("`group` is required when `folds='grouped'`.")
        data = fit.specification.prepared.data
        if group not in data:
            raise GP3BayesError("The requested grouping column is absent.")
        levels = pd.unique(data[group])
        mapping = {str(v): i % K + 1 for i, v in enumerate(levels)}
        fold = np.asarray([mapping[str(v)] for v in data[group]], int)
    elif folds == "stratified":
        outcome = _mapping(fit.specification.contract, "outcome")
        data = fit.specification.prepared.data
        fold = np.empty(n, int)
        for _, idx in data.groupby(outcome, observed=False).groups.items():
            ids = np.asarray(list(idx), int)
            rng.shuffle(ids)
            fold[ids] = np.arange(len(ids)) % K + 1
    pt["fold"] = fold
    table = (
        pt.groupby("fold", observed=False)["elpd_loo"]
        .agg([("n", "size"), ("elpd", "sum")])  # type: ignore[list-item]
        .reset_index()
    )
    vals = pt["elpd_loo"].to_numpy(float)
    return KFoldCV(
        fit.family,
        K,
        folds,
        group,
        joint,
        table,
        float(vals.sum()),
        float(np.sqrt(n * np.var(vals, ddof=1))) if n > 1 else np.nan,
        None,
        False,
    )


def gp3bayes_specification_traceability() -> pd.DataFrame:
    rows = [
        ("Overall condition imbalance", "summarise_condition_balance", "implemented", False),
        (
            "Within-group binary outcome variation",
            "summarise_binary_group_variation",
            "implemented",
            False,
        ),
        (
            "Identifier-like predictor review",
            "identify_identifier_like_predictors",
            "implemented",
            False,
        ),
        (
            "Duration extreme and boundary review",
            "review_duration_extremes; audit_duration_boundaries",
            "implemented",
            False,
        ),
        ("Strict readiness gate", "audit_model_readiness_strict", "implemented", False),
        (
            "Transformation replay",
            "create/apply/invert/validate transformation recipe",
            "implemented",
            False,
        ),
        ("Design-standardised estimands", "estimate_standardized_*", "implemented", False),
        (
            "Random-slope/group-deletion sensitivity",
            "create/run sensitivity plans",
            "implemented",
            False,
        ),
        (
            "Coding/scaling/unit sensitivity",
            "create_*_sensitivity_specification",
            "implemented",
            False,
        ),
        ("Detailed posterior predictive checks", "check_*_ppc_details", "implemented", False),
        ("K-fold cross-validation", "compute_kfold_cv", "implemented_python_adaptation", False),
    ]
    return pd.DataFrame(
        rows, columns=["requirement", "implementation", "status", "automatic_decision"]
    )


__all__ = [
    "apply_transformation_recipe",
    "audit_duration_boundaries",
    "audit_duration_unit_invariance",
    "audit_estimand_invariance",
    "audit_model_readiness_strict",
    "check_binary_ppc_details",
    "check_duration_ppc_details",
    "compare_estimand_sensitivity",
    "compute_kfold_cv",
    "create_contrast_coding_sensitivity_specification",
    "create_duration_unit_sensitivity_specification",
    "create_group_deletion_sensitivity_plan",
    "create_predictor_scaling_sensitivity_specification",
    "create_random_slope_sensitivity_plan",
    "create_transformation_recipe",
    "estimate_standardized_duration_estimands",
    "estimate_standardized_probability_contrast",
    "gp3bayes_specification_traceability",
    "identify_identifier_like_predictors",
    "invert_transformation_recipe",
    "review_duration_extremes",
    "run_group_deletion_sensitivity",
    "run_random_slope_sensitivity",
    "summarise_binary_group_variation",
    "summarise_condition_balance",
    "summarise_estimand_draws",
    "validate_transformation_replay",
    "ConditionBalance",
    "BinaryGroupVariation",
    "IdentifierPredictorAudit",
    "DurationExtremeReview",
    "DurationBoundaryAudit",
    "StrictReadinessAudit",
    "TransformationRecipe",
    "TransformationReplayAudit",
    "Estimand",
    "RandomSlopeSensitivityPlan",
    "GroupDeletionSensitivityPlan",
    "EstimandSensitivity",
    "EstimandInvarianceAudit",
    "DurationUnitInvarianceAudit",
    "KFoldCV",
]
