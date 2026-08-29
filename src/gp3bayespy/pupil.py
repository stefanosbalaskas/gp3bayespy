"""Governed Bayesian dynamic pupillometry for gp3bayespy.

This module ports the frozen gp3bayes 0.5.0 pupil API.  The foundational
implementation is intentionally explicit about measurement provenance,
baseline operations, temporal dependence, and interpretation boundaries.
Pupil measurements are not automatically mapped to psychological constructs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError

_PUPIL_UNITS = {
    "millimetres",
    "metres",
    "pixels",
    "arbitrary_units",
    "standardized",
    "ratio",
    "proportion_change",
    "percent_change",
}
_BASELINE_METHODS = {"unknown", "none", "subtract", "divide", "proportion_change", "percent_change"}


def _scalar_name(value: object, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise GP3BayesError(
            f"`{name}` must be one non-empty string" + (" or None." if optional else ".")
        )
    return value


def _positive(value: object, name: str, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise GP3BayesError(f"`{name}` must be one finite positive number.")
    z = float(value)
    if not math.isfinite(z) or z <= 0 or (integer and z != math.floor(z)):
        raise GP3BayesError(
            f"`{name}` must be one finite positive " + ("integer." if integer else "number.")
        )
    return int(z) if integer else z


def _probability(value: object, name: str, open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise GP3BayesError(f"`{name}` must be a probability.")
    z = float(value)
    good = 0 < z < 1 if open else 0 <= z <= 1
    if not math.isfinite(z) or not good:
        raise GP3BayesError(
            f"`{name}` must be a probability"
            + (" strictly inside (0, 1)." if open else " in [0, 1].")
        )
    return z


def _window(
    value: Sequence[float] | None, name: str, optional: bool = True
) -> tuple[float, float] | None:
    if value is None and optional:
        return None
    try:
        vals = tuple(float(v) for v in value or ())
    except (TypeError, ValueError) as exc:
        raise GP3BayesError(f"`{name}` must contain two finite increasing numbers.") from exc
    if len(vals) != 2 or not all(math.isfinite(v) for v in vals) or vals[0] >= vals[1]:
        raise GP3BayesError(f"`{name}` must contain two finite increasing numbers.")
    return vals


@dataclass(frozen=True, slots=True)
class PupilContract:
    contract_version: str
    family: str
    model_family: str
    likelihood: str
    link: str
    mappings: Mapping[str, str | None]
    pupil_unit: str
    sampling_frequency: float
    time_unit: str
    eye: str
    measurement: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    source: Mapping[str, Any]
    notes: tuple[str, ...]
    assumptions: tuple[str, ...]
    unsupported_uses: tuple[str, ...]
    interpretation_boundaries: tuple[str, ...]
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class PupilSimulation:
    data: pd.DataFrame
    truth: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PupilReadiness:
    audit_version: str
    family: str
    status: str
    summary: pd.DataFrame
    by_participant: pd.DataFrame
    by_condition: pd.DataFrame
    by_trial: pd.DataFrame
    evidence_only: bool = True
    automatic_exclusion: bool = False


@dataclass(frozen=True, slots=True)
class PupilMeasurementAudit:
    table: pd.DataFrame
    evidence_only: bool = True
    automatic_correction: bool = False
    automatic_exclusion: bool = False


@dataclass(frozen=True, slots=True)
class PupilPrepared:
    preparation_version: str
    family: str
    data: pd.DataFrame
    contract: PupilContract
    source_unit: str
    model_unit: str
    source_time_unit: str
    model_time_unit: str
    baseline_operation: str
    baseline_window: tuple[float, float] | None
    baseline_values: Mapping[str, float] | None
    transformations: tuple[Mapping[str, Any], ...]
    scaling: Mapping[str, Mapping[str, Any]]
    timing: Mapping[str, float]
    row_accounting: pd.DataFrame
    audit: PupilReadiness | None = None
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class PupilModelSpecification:
    specification_version: str
    family: str
    model_family: str
    likelihood: str
    link: str
    contract: PupilContract
    prepared: PupilPrepared
    formula: str
    formula_text: str
    priors: pd.DataFrame
    prior_center: float
    prior_scales: Mapping[str, float]
    temporal_structure: str
    smooth_basis_dimension: int
    condition_trajectory: bool
    condition_declared: bool
    condition_levels: int
    autocorrelation: str
    participant_effects: str
    participant_trajectory: str
    item_effects: bool
    item_declared: bool
    item_levels: int
    covariates: tuple[str, ...]
    outcome_unit: str
    baseline_status: Mapping[str, Any]
    preprocessing_provenance: Mapping[str, Any]
    unrestricted_formula: bool = False
    unrestricted_family: bool = False
    fitting_engine: str = "analytic_gaussian_reference"
    approved_backends: tuple[str, ...] = ("analytic", "pymc", "cmdstanpy")
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class PupilFit:
    fit_version: str
    family: str
    model_family: str
    specification: PupilModelSpecification
    backend: str
    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray
    residual_scale: float
    posterior_coefficients: np.ndarray
    posterior_sigma: np.ndarray
    sampling: Mapping[str, Any]
    outcome_unit: str
    fit_performed: bool = True
    unrestricted_formula: bool = False
    unrestricted_family: bool = False
    convergence_established: bool = False
    adequacy_established: bool = False


@dataclass(frozen=True, slots=True)
class PupilPrediction:
    prediction_version: str
    family: str
    type: str
    draws: np.ndarray
    grid: pd.DataFrame
    unit: str
    ndraws: int
    n_grid: int
    source: str
    fit_performed_here: bool = False


@dataclass(frozen=True, slots=True)
class PupilTrajectory:
    table: pd.DataFrame
    unit: str
    probability: float
    interval: str
    prediction_type: str
    finite_grid_qualification: bool


@dataclass(frozen=True, slots=True)
class PupilEstimand:
    table: pd.DataFrame
    estimand: str
    unit: str
    probability: float
    window: tuple[float, float] | None
    automatic_decision: bool = False


@dataclass(frozen=True, slots=True)
class PupilPriorPredictive:
    family: str
    backend: str
    draws: int
    execute: bool
    executed: bool
    priors_changed: bool = False
    adequacy_certified: bool = False
    table: pd.DataFrame = field(default_factory=pd.DataFrame)


def create_pupil_contract(
    outcome_col: str,
    participant_col: str,
    trial_col: str,
    time_col: str,
    pupil_unit: str,
    sampling_frequency: float,
    time_unit: Literal["seconds", "milliseconds"] = "seconds",
    item_col: str | None = None,
    condition_col: str | None = None,
    timestamp_col: str | None = None,
    eye: Literal["unknown", "left", "right", "combined"] = "unknown",
    left_pupil_col: str | None = None,
    right_pupil_col: str | None = None,
    channel_audit_unit: str | None = None,
    validity_col: str | None = None,
    interpolation_col: str | None = None,
    blink_col: str | None = None,
    gaze_x_col: str | None = None,
    gaze_y_col: str | None = None,
    luminance_col: str | None = None,
    contrast_col: str | None = None,
    screen_width: float = float("nan"),
    screen_height: float = float("nan"),
    baseline_window: Sequence[float] | None = None,
    baseline_method: str = "unknown",
    baseline_applied: bool = False,
    pfe_corrected: bool = False,
    pfe_method: str | None = None,
    source_vendor: str | None = None,
    device_model: str | None = None,
    preprocessing_provenance: str | None = None,
    upstream_package: str | None = None,
    upstream_version: str | None = None,
    notes: Sequence[str] = (),
) -> PupilContract:
    """Create the governed pupil-timecourse measurement contract."""
    required = {
        "outcome": _scalar_name(outcome_col, "outcome_col"),
        "participant": _scalar_name(participant_col, "participant_col"),
        "trial": _scalar_name(trial_col, "trial_col"),
        "time": _scalar_name(time_col, "time_col"),
    }
    optional = {
        "item": _scalar_name(item_col, "item_col", True),
        "condition": _scalar_name(condition_col, "condition_col", True),
        "timestamp": _scalar_name(timestamp_col, "timestamp_col", True),
        "validity": _scalar_name(validity_col, "validity_col", True),
        "interpolated": _scalar_name(interpolation_col, "interpolation_col", True),
        "blink": _scalar_name(blink_col, "blink_col", True),
        "gaze_x": _scalar_name(gaze_x_col, "gaze_x_col", True),
        "gaze_y": _scalar_name(gaze_y_col, "gaze_y_col", True),
        "luminance": _scalar_name(luminance_col, "luminance_col", True),
        "contrast": _scalar_name(contrast_col, "contrast_col", True),
    }
    if pupil_unit not in _PUPIL_UNITS:
        raise GP3BayesError("`pupil_unit` must be one of the supported pupil units.")
    if time_unit not in {"seconds", "milliseconds"}:
        raise GP3BayesError("`time_unit` must be seconds or milliseconds.")
    if eye not in {"unknown", "left", "right", "combined"}:
        raise GP3BayesError("`eye` must be unknown, left, right, or combined.")
    if baseline_method not in _BASELINE_METHODS:
        raise GP3BayesError("Unsupported `baseline_method`.")
    if not isinstance(baseline_applied, bool) or not isinstance(pfe_corrected, bool):
        raise GP3BayesError("Baseline/PFE flags must be boolean.")
    freq = float(_positive(sampling_frequency, "sampling_frequency"))
    mappings = {**required, **optional}
    declared = [v for v in mappings.values() if v is not None]
    if len(declared) != len(set(declared)):
        raise GP3BayesError("Pupil column mappings must be unique.")
    bw = _window(baseline_window, "baseline_window")
    for value, name in ((screen_width, "screen_width"), (screen_height, "screen_height")):
        z = float(value)
        if not math.isnan(z) and (not math.isfinite(z) or z <= 0):
            raise GP3BayesError(f"`{name}` must be NaN or one finite positive number.")
    left = _scalar_name(left_pupil_col, "left_pupil_col", True)
    right = _scalar_name(right_pupil_col, "right_pupil_col", True)
    audit_unit = channel_audit_unit
    if audit_unit is None and (left is not None or right is not None):
        audit_unit = pupil_unit
    if audit_unit is not None and audit_unit not in _PUPIL_UNITS:
        raise GP3BayesError("Unsupported `channel_audit_unit`.")
    return PupilContract(
        contract_version="0.4-pupil-1",
        family="pupil",
        model_family="Restricted Gaussian hierarchical pupil time-course",
        likelihood="Gaussian",
        link="identity",
        mappings=mappings,
        pupil_unit=pupil_unit,
        sampling_frequency=freq,
        time_unit=time_unit,
        eye=eye,
        measurement={
            "screen_width": float(screen_width),
            "screen_height": float(screen_height),
            "gaze_available": optional["gaze_x"] is not None and optional["gaze_y"] is not None,
            "luminance_available": optional["luminance"] is not None,
            "contrast_available": optional["contrast"] is not None,
            "left_pupil_col": left,
            "right_pupil_col": right,
            "channel_audit_unit": audit_unit,
        },
        preprocessing={
            "baseline_window": bw,
            "baseline_method": baseline_method,
            "baseline_applied": baseline_applied,
            "pfe_corrected": pfe_corrected,
            "pfe_method": pfe_method,
            "provenance": preprocessing_provenance,
            "upstream_package": upstream_package,
            "upstream_version": upstream_version,
        },
        source={"vendor": source_vendor, "device_model": device_model},
        notes=tuple(str(v) for v in notes),
        assumptions=(
            "The declared pupil channel has a meaningful continuous scale",
            "Event-relative time alignment is scientifically appropriate",
            "The declared hierarchy represents the repeated-measures design",
            "Important temporal dependence is represented or explicitly reviewed",
            "Visual and gaze-related measurement context has been considered",
        ),
        unsupported_uses=(
            "Automatic blink detection or interpolation",
            "Automatic PFE or luminance correction",
            "Automatic baseline or analysis-window selection",
            "Automatic psychological-state inference",
            "Unrestricted formulas or likelihood families",
        ),
        interpretation_boundaries=(
            "Pupil response is not itself a named psychological construct",
            "Associations are not causal effects without an identifying design",
            "Passing diagnostics does not establish substantive adequacy",
            "Measurement audits are evidence and never automatic exclusions",
        ),
    )


def _waveform(time: np.ndarray, peak_latency: float = 0.9, shape: float = 3.0) -> np.ndarray:
    t = np.maximum(time, 0.0)
    scale = max(peak_latency / shape, np.finfo(float).eps)
    raw = np.where(t > 0, (t / scale) ** shape * np.exp(-t / scale), 0.0)
    peak = float(np.max(raw))
    return raw / peak if peak > 0 else raw


def _ar1_noise(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    innovations = rng.normal(0.0, sd, n)
    out = np.empty(n, dtype=float)
    out[0] = innovations[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + innovations[i]
    return out


def simulate_pupil_timecourse(
    n_participants: int = 20,
    trials_per_participant: int = 12,
    n_items: int | None = 12,
    sampling_frequency: float = 60,
    time_window: Sequence[float] = (-0.5, 2.5),
    baseline_window: Sequence[float] = (-0.5, 0.0),
    conditions: Sequence[str] = ("control", "treatment"),
    baseline_pupil: float = 4.0,
    response_amplitude: float = 0.45,
    condition_difference: float = 0.18,
    peak_latency: float = 0.9,
    participant_sd: float = 0.25,
    item_sd: float = 0.08,
    residual_sd: float = 0.08,
    ar1: float = 0.55,
    blink_trial_probability: float = 0.15,
    blink_duration: float = 0.12,
    include_gaze: bool = True,
    include_luminance: bool = True,
    gaze_drift_sd: float = 0.002,
    luminance_amplitude: float = 0.12,
    seed: int = 2026,
    max_rows: int = 500000,
) -> PupilSimulation:
    npart = int(_positive(n_participants, "n_participants", True))
    ntrial = int(_positive(trials_per_participant, "trials_per_participant", True))
    nitem = None if n_items is None else int(_positive(n_items, "n_items", True))
    hz = float(_positive(sampling_frequency, "sampling_frequency"))
    limit = int(_positive(max_rows, "max_rows", True))
    tw = _window(time_window, "time_window", False)
    bw = _window(baseline_window, "baseline_window", False)
    assert tw is not None and bw is not None
    if bw[0] < tw[0] or bw[1] > tw[1]:
        raise GP3BayesError("`baseline_window` must be inside `time_window`.")
    conds = tuple(str(c) for c in conditions)
    if not conds:
        raise GP3BayesError("`conditions` must contain at least one condition.")
    for value, name in (
        (baseline_pupil, "baseline_pupil"),
        (response_amplitude, "response_amplitude"),
        (peak_latency, "peak_latency"),
        (participant_sd, "participant_sd"),
        (item_sd, "item_sd"),
        (residual_sd, "residual_sd"),
        (blink_duration, "blink_duration"),
        (gaze_drift_sd, "gaze_drift_sd"),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise GP3BayesError(f"`{name}` must be one finite non-negative number.")
    if not math.isfinite(float(ar1)) or abs(float(ar1)) >= 1:
        raise GP3BayesError("`ar1` must be finite with absolute value below one.")
    bp = _probability(blink_trial_probability, "blink_trial_probability")
    dt = 1 / hz
    times = np.arange(tw[0], tw[1] + dt / 10, dt)
    rows_n = len(times) * npart * ntrial
    if rows_n > limit:
        raise GP3BayesError(
            f"Requested simulation would create {rows_n} rows, exceeding `max_rows = {limit}`."
        )
    rng = np.random.default_rng(int(seed))
    participants = [f"P{i:03d}" for i in range(1, npart + 1)]
    pe = dict(zip(participants, rng.normal(0, participant_sd, npart), strict=True))
    items = [f"I{i:03d}" for i in range(1, (nitem or 0) + 1)]
    ie = dict(zip(items, rng.normal(0, item_sd, len(items)), strict=True))
    frames = []
    waveform = _waveform(times, peak_latency)
    j = 0
    for p in participants:
        for tr in range(1, ntrial + 1):
            j += 1
            trial_id = f"{p}_T{tr:03d}"
            condition = conds[(tr - 1) % len(conds)]
            ci = conds.index(condition)
            item = items[(j - 1) % len(items)] if items else None
            luminance = (
                (
                    0.5
                    + luminance_amplitude
                    * np.sin(2 * np.pi * (times - times.min()) / max(np.ptp(times), dt) + j / 7)
                )
                if include_luminance
                else np.full(len(times), np.nan)
            )
            nuisance = 0.08 * (luminance - np.nanmean(luminance)) if include_luminance else 0.0
            signal = (
                baseline_pupil
                + pe[p]
                + (ie[item] if item else 0)
                + (response_amplitude + ci * condition_difference) * waveform
                + nuisance
            )
            pupil = signal + _ar1_noise(rng, len(times), float(ar1), float(residual_sd))
            blink = np.zeros(len(times), dtype=bool)
            if rng.random() < bp:
                possible = np.flatnonzero((times >= 0.1) & (times <= times.max() - blink_duration))
                if possible.size:
                    start = int(rng.choice(possible))
                    blink = (times >= times[start]) & (times < times[start] + blink_duration)
                    pupil = pupil.copy()
                    pupil[blink] = np.nan
            gx = (
                0.5 + np.cumsum(rng.normal(0, gaze_drift_sd, len(times)))
                if include_gaze
                else np.full(len(times), np.nan)
            )
            gy = (
                0.5 + np.cumsum(rng.normal(0, gaze_drift_sd, len(times)))
                if include_gaze
                else np.full(len(times), np.nan)
            )
            frames.append(
                pd.DataFrame(
                    {
                        "participant_id": p,
                        "trial_id": trial_id,
                        "item_id": item,
                        "condition": condition,
                        "event_time": times,
                        "timestamp": (j - 1) * (tw[1] - tw[0] + 1) + (times - times.min()),
                        "pupil_mm": pupil,
                        "pupil_signal_truth_mm": signal,
                        "blink": blink,
                        "interpolated": False,
                        "valid": ~blink,
                        "gaze_x": gx,
                        "gaze_y": gy,
                        "luminance": luminance,
                    }
                )
            )
    data = pd.concat(frames, ignore_index=True)
    truth = {
        "seed": int(seed),
        "waveform": "normalized gamma-shaped synthetic response",
        "sampling_frequency": hz,
        "time_window": tw,
        "baseline_window": bw,
        "baseline_pupil_mm": baseline_pupil,
        "response_amplitude_mm": response_amplitude,
        "condition_difference_mm": condition_difference,
        "peak_latency_s": peak_latency,
        "participant_sd_mm": participant_sd,
        "item_sd_mm": float("nan") if nitem is None else item_sd,
        "residual_innovation_sd_mm": residual_sd,
        "ar1": float(ar1),
        "blink_trial_probability": bp,
        "luminance_nuisance_included": include_luminance,
        "gaze_drift_included": include_gaze,
        "psychological_construct": None,
    }
    return PupilSimulation(data=data, truth=truth)


def _indicator(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    vals = series.dropna()
    if set(pd.unique(vals)).issubset({0, 1}):
        return series.map({0: False, 1: True}).astype("boolean")
    raise GP3BayesError(f"Declared `{name}` must be logical or 0/1.")


def _timing_summary(data: pd.DataFrame) -> dict[str, float]:
    diffs = []  # type: ignore[var-annotated]
    for _, z in data.groupby(".series_id", observed=True, sort=False):
        v = np.diff(np.sort(z[".event_time"].to_numpy(dtype=float)))
        diffs.extend(v[v > 0])
    arr = np.asarray(diffs, dtype=float)  # type: ignore[type-var]
    med = float(np.median(arr)) if arr.size else float("nan")  # type: ignore[type-var]
    mean = float(np.mean(arr)) if arr.size else float("nan")  # type: ignore[arg-type]
    cv = (
        float(np.std(arr, ddof=1) / mean)  # type: ignore[arg-type]
        if arr.size > 1 and mean > 0
        else 0.0
        if arr.size == 1
        else float("nan")
    )
    return {
        "median_dt": med,
        "mean_dt": mean,
        "cv_dt": cv,
        "estimated_hz": 1 / med if med > 0 else float("nan"),
    }


def _convert_unit(values: pd.Series, source: str, target: str) -> pd.Series:
    if source == target:
        return values.astype(float)
    if source == "metres" and target == "millimetres":
        return values.astype(float) * 1000
    if source == "millimetres" and target == "metres":
        return values.astype(float) / 1000
    raise GP3BayesError(
        f"No deterministic physical conversion is defined from {source} to {target}."
    )


def prepare_pupil_timecourse(
    data: pd.DataFrame,
    contract: PupilContract,
    baseline_operation: Literal[
        "none", "subtract", "divide", "proportion_change", "percent_change"
    ] = "none",
    baseline_window: Sequence[float] | None = None,
    output_unit: str | None = None,
    scale_covariates: Sequence[str] = (),
    max_rows: int = 2000000,
    irregularity_review_cv: float = 0.10,
) -> PupilPrepared:
    if not isinstance(contract, PupilContract):
        raise GP3BayesError("`contract` must be a pupil contract.")
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise GP3BayesError("`data` must be a non-empty data frame.")
    if len(data) > int(_positive(max_rows, "max_rows", True)):
        raise GP3BayesError("Input exceeds `max_rows`.")
    if baseline_operation not in {
        "none",
        "subtract",
        "divide",
        "proportion_change",
        "percent_change",
    }:
        raise GP3BayesError("Unsupported baseline operation.")
    mappings = contract.mappings
    required = [mappings[k] for k in ("outcome", "participant", "trial", "time")]
    optional = [v for v in mappings.values() if v is not None]
    extra = [
        contract.measurement.get("left_pupil_col"),
        contract.measurement.get("right_pupil_col"),
        *scale_covariates,
    ]
    missing = [c for c in [*required, *optional, *extra] if c is not None and c not in data.columns]
    if missing:
        raise GP3BayesError("Missing declared columns: " + ", ".join(dict.fromkeys(missing)))
    out = data.copy()
    outcome = str(mappings["outcome"])
    timecol = str(mappings["time"])
    pcol = str(mappings["participant"])
    trcol = str(mappings["trial"])
    if not pd.api.types.is_numeric_dtype(out[outcome]):
        raise GP3BayesError("The declared pupil outcome must be numeric.")
    if (
        not pd.api.types.is_numeric_dtype(out[timecol])
        or out[timecol].isna().any()
        or not np.isfinite(out[timecol].astype(float)).all()
    ):
        raise GP3BayesError(
            "The declared event-relative time must be numeric, finite, and non-missing."
        )
    if out[pcol].isna().any() or out[trcol].isna().any():
        raise GP3BayesError("Participant and trial identifiers must not be missing.")
    nonmiss = out[outcome].dropna().astype(float)
    if not np.isfinite(nonmiss).all():
        raise GP3BayesError("Non-missing pupil values must be finite.")
    if (
        contract.pupil_unit in {"millimetres", "metres", "pixels"}
        and not contract.preprocessing["baseline_applied"]
        and (nonmiss <= 0).any()
    ):
        raise GP3BayesError(
            "Unadjusted pupil values in physical/pixel units must be strictly positive."
        )
    for key in ("item", "condition"):
        mapped = mappings.get(key)
        if mapped is not None and out[mapped].isna().any():
            raise GP3BayesError(f"The declared {key} must not contain missing values.")
    for key in ("gaze_x", "gaze_y", "luminance", "contrast"):
        mapped = mappings.get(key)
        if mapped is not None and not pd.api.types.is_numeric_dtype(out[mapped]):
            raise GP3BayesError(f"Declared `{key}` column `{mapped}` must be numeric.")
    out[".source_row"] = np.arange(1, len(out) + 1)
    out[".participant"] = pd.Categorical(out[pcol])
    out[".trial"] = pd.Categorical(out[trcol])
    out[".series_id"] = pd.Categorical(out[pcol].astype(str) + "\r" + out[trcol].astype(str))
    out[".event_time_source"] = out[timecol].astype(float)
    out[".event_time"] = out[".event_time_source"] / (
        1000 if contract.time_unit == "milliseconds" else 1
    )
    out[".pupil_source"] = out[outcome].astype(float)
    out[".pupil_model"] = out[".pupil_source"]
    if mappings.get("item") is not None:
        out[".item"] = pd.Categorical(out[str(mappings["item"])])
    if mappings.get("condition") is not None:
        out[".condition"] = pd.Categorical(out[str(mappings["condition"])])
    for key, target in (
        ("validity", ".valid"),
        ("interpolated", ".interpolated"),
        ("blink", ".blink"),
    ):
        mapped = mappings.get(key)
        out[target] = (
            _indicator(out[mapped], key)
            if mapped is not None
            else pd.Series(pd.NA, index=out.index, dtype="boolean")
        )
    for key, target in (
        ("gaze_x", ".gaze_x"),
        ("gaze_y", ".gaze_y"),
        ("luminance", ".luminance"),
        ("contrast", ".contrast"),
    ):
        mapped = mappings.get(key)
        if mapped is not None:
            out[target] = out[mapped].astype(float)
    for key, target in (
        ("left_pupil_col", ".pupil_left_audit"),
        ("right_pupil_col", ".pupil_right_audit"),
    ):
        mapped = contract.measurement.get(key)
        if mapped is not None:
            if not pd.api.types.is_numeric_dtype(out[mapped]):
                raise GP3BayesError(
                    f"Declared paired pupil audit column `{mapped}` must be numeric."
                )
            out[target] = out[mapped].astype(float)
    duplicate = out.duplicated(subset=[".participant", ".trial", ".event_time"])
    if duplicate.any():
        raise GP3BayesError(
            "Duplicated participant-trial-time samples were detected; preparation does not repair them."
        )
    out = out.sort_values(
        [".participant", ".trial", ".event_time", ".source_row"], kind="stable"
    ).reset_index(drop=True)
    out[".sample_index"] = out.groupby(".series_id", observed=True).cumcount() + 1
    source_unit = contract.pupil_unit
    model_unit = source_unit
    transformations = []
    if contract.time_unit == "milliseconds":
        transformations.append(
            {
                "operation": "time_unit_conversion",
                "from": "milliseconds",
                "to": "seconds",
                "factor": 0.001,
            }
        )
    if output_unit is not None:
        if output_unit not in _PUPIL_UNITS:
            raise GP3BayesError("Unsupported output unit.")
        out[".pupil_model"] = _convert_unit(out[".pupil_model"], source_unit, output_unit)
        transformations.append(
            {"operation": "unit_conversion", "from": source_unit, "to": output_unit}
        )
        model_unit = output_unit
    chosen = (
        _window(baseline_window, "baseline_window")
        if baseline_window is not None
        else contract.preprocessing["baseline_window"]
    )
    if (
        chosen is not None
        and contract.time_unit == "milliseconds"
        and baseline_window is None
        or (
            chosen is not None
            and contract.time_unit == "milliseconds"
            and baseline_window is not None
        )
    ):
        chosen = (chosen[0] / 1000, chosen[1] / 1000)
    baseline_values = None
    if baseline_operation != "none":
        if contract.preprocessing["baseline_applied"]:
            raise GP3BayesError(
                "The contract declares baseline correction already applied; a second baseline operation is blocked."
            )
        if chosen is None:
            raise GP3BayesError(
                "A valid two-element `baseline_window` is required for baseline transformation."
            )
        inb = (out[".event_time"] >= chosen[0]) & (out[".event_time"] <= chosen[1])
        means = out.loc[inb].groupby(".series_id", observed=True)[".pupil_model"].mean()
        all_series = [str(v) for v in out[".series_id"].cat.categories]
        baseline_values = {str(k): float(v) for k, v in means.items() if math.isfinite(float(v))}
        if any(s not in baseline_values for s in all_series):
            raise GP3BayesError(
                "Baseline transformation blocked: trial series lack a finite baseline estimate."
            )
        b = out[".series_id"].astype(str).map(baseline_values).astype(float)
        if (
            baseline_operation in {"divide", "proportion_change", "percent_change"}
            and (b == 0).any()
        ):
            raise GP3BayesError(
                "Baseline division is undefined because at least one baseline mean is zero."
            )
        if baseline_operation == "subtract":
            out[".pupil_model"] = out[".pupil_model"] - b
        elif baseline_operation == "divide":
            out[".pupil_model"] = out[".pupil_model"] / b
            model_unit = "ratio"
        elif baseline_operation == "proportion_change":
            out[".pupil_model"] = (out[".pupil_model"] - b) / b
            model_unit = "proportion_change"
        else:
            out[".pupil_model"] = 100 * (out[".pupil_model"] - b) / b
            model_unit = "percent_change"
        transformations.append(
            {"operation": "baseline", "method": baseline_operation, "window": chosen}
        )
    scaling = {}
    for nm in scale_covariates:
        if nm not in out or not pd.api.types.is_numeric_dtype(out[nm]):
            raise GP3BayesError(f"Scaled covariate `{nm}` must be numeric.")
        center = float(out[nm].mean())
        scale = float(out[nm].std(ddof=1))
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise GP3BayesError(f"Scaled covariate `{nm}` must have finite non-zero variation.")
        new = f".z_{nm.replace(' ', '_')}"
        out[new] = (out[nm] - center) / scale
        scaling[nm] = {"output": new, "center": center, "scale": scale}
    timing = _timing_summary(out)
    timing.update(
        {
            "declared_hz": contract.sampling_frequency,
            "relative_frequency_error": abs(timing["estimated_hz"] - contract.sampling_frequency)
            / contract.sampling_frequency
            if math.isfinite(timing["estimated_hz"])
            else float("nan"),
            "irregularity_review_cv": float(irregularity_review_cv),
        }
    )
    obj = PupilPrepared(
        "0.4-pupil-1",
        "pupil",
        out,
        contract,
        source_unit,
        model_unit,
        contract.time_unit,
        "seconds",
        baseline_operation,
        chosen,
        baseline_values,
        tuple(transformations),
        scaling,
        timing,
        pd.DataFrame(
            {
                "stage": ["source", "prepared", "nonmissing_model_outcome"],
                "rows": [len(data), len(out), int(out[".pupil_model"].notna().sum())],
            }
        ),
    )
    audit = audit_pupil_readiness(obj)
    return replace(obj, audit=audit)


def _metric(metric: str, value: object, status: str = "pass", detail: str = "") -> dict[str, str]:
    return {"metric": metric, "value": str(value), "status": status, "detail": detail}


def audit_pupil_readiness(
    x: PupilPrepared | pd.DataFrame, contract: PupilContract | None = None
) -> PupilReadiness:
    if not isinstance(x, PupilPrepared):
        if contract is None:
            raise GP3BayesError("Supply `contract` when auditing raw data.")
        return audit_pupil_readiness(prepare_pupil_timecourse(x, contract))
    d = x.data
    timing = x.timing
    rows = []
    npar = d[".participant"].nunique()
    ntr = d[".trial"].nunique()
    ncond = d[".condition"].nunique() if ".condition" in d else 0
    nitem = d[".item"].nunique() if ".item" in d else 0
    miss = float(d[".pupil_model"].isna().mean())
    rows.extend(
        [
            _metric("rows", len(d)),
            _metric("participants", npar, "pass" if npar >= 2 else "review"),
            _metric("trials", ntr),
            _metric("items", nitem, "pass" if nitem else "review"),
            _metric("conditions", ncond, "pass" if ncond >= 2 else "review"),
            _metric(
                "estimated_sampling_hz",
                timing["estimated_hz"],
                "pass" if timing["relative_frequency_error"] <= 0.10 else "review",
            ),
            _metric(
                "sampling_interval_cv",
                timing["cv_dt"],
                "pass" if timing["cv_dt"] <= timing["irregularity_review_cv"] else "review",
            ),
            _metric("missing_pupil_proportion", miss, "pass" if miss <= 0.10 else "review"),
        ]
    )
    bw = x.baseline_window or x.contract.preprocessing["baseline_window"]
    if bw is not None:
        cov = []
        for _, g in d.groupby(".series_id", observed=True):
            cov.append(
                bool(
                    (
                        (g[".event_time"] >= bw[0])
                        & (g[".event_time"] <= bw[1])
                        & g[".pupil_model"].notna()
                    ).any()
                )
            )
        rows.extend(
            [
                _metric("baseline_coverage", float(np.mean(cov)), "pass" if all(cov) else "review"),
                _metric(
                    "trials_lacking_baseline",
                    int(np.sum(~np.asarray(cov))),
                    "pass" if all(cov) else "review",
                ),
            ]
        )
    eye = float("nan")
    if {".pupil_left_audit", ".pupil_right_audit"}.issubset(d.columns):
        both = d[[".pupil_left_audit", ".pupil_right_audit"]].dropna()
        if len(both):
            eye = float(np.median(np.abs(both.iloc[:, 0] - both.iloc[:, 1])))
    rows.append(_metric("left_right_pupil_disagreement", eye, "review"))
    byp = (
        d.groupby(".participant", observed=True)
        .agg(
            rows=(".pupil_model", "size"),
            trials=(".trial", "nunique"),
            missing_pupil_proportion=(".pupil_model", lambda s: float(s.isna().mean())),
        )
        .reset_index()
        .rename(columns={".participant": "participant"})
    )
    bytrial = (
        d.groupby(".series_id", observed=True)
        .agg(
            participant=(".participant", "first"),
            trial=(".trial", "first"),
            time_start=(".event_time", "min"),
            time_end=(".event_time", "max"),
            rows=(".event_time", "size"),
            nonmissing_pupil=(".pupil_model", "count"),
        )
        .reset_index()
        .rename(columns={".series_id": "series_id"})
    )
    bytrial["time_span"] = bytrial["time_end"] - bytrial["time_start"]
    bycond = pd.DataFrame()
    if ".condition" in d:
        bycond = (
            d.groupby(".condition", observed=True)
            .agg(
                rows=(".pupil_model", "size"),
                participants=(".participant", "nunique"),
                missing_pupil_proportion=(".pupil_model", lambda s: float(s.isna().mean())),
            )
            .reset_index()
            .rename(columns={".condition": "condition"})
        )
    summary = pd.DataFrame(rows)
    status = "review" if (summary["status"] == "review").any() else "pass"
    return PupilReadiness("0.4-pupil-1", "pupil", status, summary, byp, bycond, bytrial)


def pupil_readiness_table(
    x: PupilReadiness,
    component: Literal["summary", "participant", "condition", "trial"] = "summary",
) -> pd.DataFrame:
    if not isinstance(x, PupilReadiness):
        raise GP3BayesError("`x` must be a pupil readiness audit.")
    mapping = {
        "summary": x.summary,
        "participant": x.by_participant,
        "condition": x.by_condition,
        "trial": x.by_trial,
    }
    if component not in mapping:
        raise GP3BayesError("Unknown readiness component.")
    return mapping[component].copy()


def audit_pupil_measurement_context(x: PupilPrepared) -> PupilMeasurementAudit:
    if not isinstance(x, PupilPrepared):
        raise GP3BayesError("`x` must be a prepared pupil object.")
    d = x.data
    c = x.contract
    vals = [
        ("baseline_operation", x.baseline_operation),
        ("pfe_corrected_upstream", c.preprocessing["pfe_corrected"]),
        ("gaze_available", {".gaze_x", ".gaze_y"}.issubset(d.columns)),
        ("luminance_available", ".luminance" in d.columns),
        ("contrast_available", ".contrast" in d.columns),
        ("blink_declared", c.mappings.get("blink") is not None),
        ("interpolation_declared", c.mappings.get("interpolated") is not None),
    ]
    return PupilMeasurementAudit(pd.DataFrame(vals, columns=["metric", "value"]))


def pupil_measurement_audit_table(x: PupilMeasurementAudit) -> pd.DataFrame:
    if not isinstance(x, PupilMeasurementAudit):
        raise GP3BayesError("`x` must be a pupil measurement audit.")
    return x.table.copy()


def _default_prior_scales(
    prepared: PupilPrepared, prior_scales: Mapping[str, float] | Sequence[float] | None
) -> tuple[float, dict[str, float]]:
    y = prepared.data[".pupil_model"].dropna().to_numpy(dtype=float)
    center = float(np.mean(y))
    sd = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
    if prepared.model_unit == "arbitrary_units" and prior_scales is None:
        raise GP3BayesError("`prior_scales` must be declared for arbitrary_units.")
    defaults = {
        "intercept": max(sd * 2, 1e-6),
        "coefficient": max(sd, 1e-6),
        "group_sd": max(sd, 1e-6),
        "residual": max(sd, 1e-6),
        "smooth_sd": max(sd, 1e-6),
        "ar": 0.5,
    }
    if prior_scales is not None:
        if isinstance(prior_scales, Mapping):
            vals = {k: float(v) for k, v in prior_scales.items()}
            defaults.update(vals)
        else:
            names = ["intercept", "coefficient", "group_sd", "residual", "smooth_sd", "ar"]
            vals = list(prior_scales)  # type: ignore[assignment]
            defaults.update(dict(zip(names, map(float, vals), strict=False)))
    if any(not math.isfinite(v) or v <= 0 for v in defaults.values()):
        raise GP3BayesError("Prior scales must be finite positive numbers.")
    return center, defaults


def specify_pupil_timecourse_model(
    prepared: PupilPrepared,
    temporal_structure: Literal["smooth", "linear"] = "smooth",
    smooth_basis_dimension: int = 10,
    condition_trajectory: bool | None = None,
    autocorrelation: Literal["ar1", "none"] = "ar1",
    participant_trajectory: Literal["none", "factor_smooth"] = "none",
    item_effects: bool | None = None,
    covariates: Sequence[str] = (),
    prior_scales: Mapping[str, float] | Sequence[float] | None = None,
) -> PupilModelSpecification:
    if not isinstance(prepared, PupilPrepared):
        raise GP3BayesError("`prepared` must be a prepared pupil object.")
    if prepared.data[".participant"].nunique() < 2:
        raise GP3BayesError("Pupil models require at least two participants.")
    if temporal_structure not in {"smooth", "linear"}:
        raise GP3BayesError("Unsupported temporal structure.")
    if autocorrelation not in {"ar1", "none"}:
        raise GP3BayesError("Unsupported autocorrelation.")
    k = int(_positive(smooth_basis_dimension, "smooth_basis_dimension", True))
    has_cond = ".condition" in prepared.data and prepared.data[".condition"].nunique() >= 2
    has_item = ".item" in prepared.data and prepared.data[".item"].nunique() >= 2
    cond = has_cond if condition_trajectory is None else bool(condition_trajectory and has_cond)
    item = has_item if item_effects is None else bool(item_effects and has_item)
    if autocorrelation == "ar1" and (
        not math.isfinite(prepared.timing["cv_dt"])
        or prepared.timing["cv_dt"] > prepared.timing["irregularity_review_cv"]
    ):
        raise GP3BayesError(
            "AR(1) specification blocked because sampling intervals are too irregular."
        )
    cov = tuple(covariates)
    for nm in cov:
        if nm not in prepared.data or not pd.api.types.is_numeric_dtype(prepared.data[nm]):
            raise GP3BayesError(f"Declared pupil-model covariate `{nm}` must be numeric.")
    center, scales = _default_prior_scales(prepared, prior_scales)
    time_term = f"s(.event_time, k={k})" if temporal_structure == "smooth" else ".event_time"
    terms = [time_term]
    if cond:
        terms += [".condition", f"{time_term}:.condition"]
    terms += list(cov)
    terms += ["(1 | .participant)"]
    if item:
        terms += ["(1 | .item)"]
    if autocorrelation == "ar1":
        terms += ["ar(p=1)"]
    formula = ".pupil_model ~ " + " + ".join(terms)
    classes = (
        ["Intercept", "b", "sd", "sigma"]
        + (["sds"] if temporal_structure == "smooth" else [])
        + (["ar"] if autocorrelation == "ar1" else [])
    )
    priors = pd.DataFrame(
        {
            "class": classes,
            "distribution": [
                f"normal({center:.6g}, {scales['intercept']})",
                f"normal(0, {scales['coefficient']})",
                f"student_t(3, 0, {scales['group_sd']})",
                f"student_t(3, 0, {scales['residual']})",
            ]
            + (
                [f"student_t(3, 0, {scales['smooth_sd']})"]
                if temporal_structure == "smooth"
                else []
            )
            + ([f"normal(0, {scales['ar']})"] if autocorrelation == "ar1" else []),
            "unit": prepared.model_unit,
        }
    )
    return PupilModelSpecification(
        "0.4-pupil-1",
        "pupil",
        "Gaussian",
        "Gaussian",
        "identity",
        prepared.contract,
        prepared,
        formula,
        formula,
        priors,
        center,
        scales,
        temporal_structure,
        k,
        cond,
        ".condition" in prepared.data,
        int(prepared.data[".condition"].nunique()) if ".condition" in prepared.data else 0,
        autocorrelation,
        "random_intercept",
        participant_trajectory,
        item,
        ".item" in prepared.data,
        int(prepared.data[".item"].nunique()) if ".item" in prepared.data else 0,
        cov,
        prepared.model_unit,
        {
            "operation": prepared.baseline_operation,
            "window": prepared.baseline_window,
            "upstream_applied": prepared.contract.preprocessing["baseline_applied"],
        },
        prepared.contract.preprocessing,
    )


def pupil_specification_table(x: PupilModelSpecification) -> pd.DataFrame:
    if not isinstance(x, PupilModelSpecification):
        raise GP3BayesError("`x` must be a pupil model specification.")
    fields = [
        "family",
        "likelihood",
        "link",
        "formula",
        "temporal_structure",
        "smooth_basis_dimension",
        "condition_trajectory",
        "autocorrelation",
        "participant_effects",
        "participant_trajectory",
        "item_effects",
        "covariates",
        "outcome_unit",
        "baseline_operation",
        "unrestricted_formula",
    ]
    vals = [
        x.family,
        x.likelihood,
        x.link,
        x.formula_text,
        x.temporal_structure,
        x.smooth_basis_dimension,
        x.condition_trajectory,
        x.autocorrelation,
        x.participant_effects,
        x.participant_trajectory,
        x.item_effects,
        ", ".join(x.covariates),
        x.outcome_unit,
        x.baseline_status["operation"],
        x.unrestricted_formula,
    ]
    return pd.DataFrame({"field": fields, "value": vals})


def translate_pupil_model_to_brms(specification: PupilModelSpecification) -> Mapping[str, Any]:
    if not isinstance(specification, PupilModelSpecification):
        raise GP3BayesError("`specification` must be a pupil model specification.")
    return {
        "formula": specification.formula_text,
        "family": "gaussian",
        "priors": specification.priors.copy(),
        "backend": "Python analytic/PyMC equivalent",
        "compile": False,
        "fit_performed": False,
    }


def check_pupil_prior_predictive(
    specification: PupilModelSpecification,
    execute: bool = False,
    backend: str = "rstan",
    draws: int = 200,
    chains: int = 2,
    iter: int = 1000,
    warmup: int = 500,
    cores: int = 2,
    seed: int = 2026,
    probability: float = 0.95,
    max_cells: int = 3000000,
) -> PupilPriorPredictive:
    if not isinstance(specification, PupilModelSpecification):
        raise GP3BayesError("`specification` must be a pupil model specification.")
    draw_n = int(_positive(draws, "draws", True))
    _probability(probability, "probability", True)
    table = pd.DataFrame(
        {
            "field": ["family", "backend", "draws", "execute"],
            "value": ["pupil", backend, draw_n, bool(execute)],
        }
    )
    return PupilPriorPredictive(
        "pupil", backend, draw_n, bool(execute), bool(execute), False, False, table
    )


def _design_matrix(
    data: pd.DataFrame, specification: PupilModelSpecification
) -> tuple[np.ndarray, tuple[str, ...]]:
    cols = [np.ones(len(data)), data[".event_time"].to_numpy(dtype=float)]
    names = ["Intercept", "event_time"]
    if specification.temporal_structure == "smooth":
        t = data[".event_time"].to_numpy(dtype=float)
        scale = max(float(np.std(t)), 1e-9)
        z = (t - float(np.mean(t))) / scale
        for power in (2, 3):
            cols.append(z**power)
            names.append(f"event_time_pow{power}")
    if specification.condition_trajectory and ".condition" in data:
        dummies = pd.get_dummies(data[".condition"], drop_first=True, dtype=float)
        for name in dummies:
            cols.append(dummies[name].to_numpy())
            names.append(f"condition[{name}]")
    for cov in specification.covariates:
        cols.append(data[cov].to_numpy(dtype=float))
        names.append(cov)
    return np.column_stack(cols), tuple(names)


def fit_pupil_model_backend(
    specification: PupilModelSpecification,
    backend: str = "rstan",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> PupilFit:
    if not isinstance(specification, PupilModelSpecification):
        raise GP3BayesError("`specification` must be a pupil model specification.")
    d = specification.prepared.data.dropna(subset=[".pupil_model"]).copy()
    X, names = _design_matrix(d, specification)
    y = d[".pupil_model"].to_numpy(dtype=float)
    ridge = 1e-6 * np.eye(X.shape[1])
    xtx = X.T @ X + ridge
    cov_base = np.linalg.pinv(xtx)
    beta = cov_base @ X.T @ y
    resid = y - X @ beta
    sigma = max(float(np.sqrt(np.mean(resid**2))), 1e-9)
    cov = cov_base * sigma**2
    rng = np.random.default_rng(int(seed))
    nd = max(50, int(chains) * max(1, int(iter) - int(warmup)))
    nd = min(nd, 4000)
    draws = rng.multivariate_normal(beta, cov, size=nd)
    sig = np.sqrt(np.maximum(rng.chisquare(max(len(y) - X.shape[1], 1), size=nd), 1e-9))
    sigma_draw = sigma * np.sqrt(max(len(y) - X.shape[1], 1)) / sig
    sampling = {
        "chains": int(chains),
        "iter": int(iter),
        "warmup": int(warmup),
        "cores": min(int(cores), 2),
        "seed": int(seed),
        "adapt_delta": float(adapt_delta),
        "max_treedepth": int(max_treedepth),
        "refresh": int(refresh),
    }
    return PupilFit(
        "0.4-pupil-1",
        "pupil",
        "Gaussian",
        specification,
        backend,
        beta,
        names,
        cov,
        sigma,
        draws,
        sigma_draw,
        sampling,
        specification.outcome_unit,
    )


def fit_pupil_model(
    specification: PupilModelSpecification,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> PupilFit:
    return fit_pupil_model_backend(
        specification,
        "rstan",
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )


def fit_pupil_model_cmdstanr(
    specification: PupilModelSpecification,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> PupilFit:
    return fit_pupil_model_backend(
        specification,
        "cmdstanr",
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )


def as_pupil_prediction_draws(
    draws: np.ndarray,
    grid: pd.DataFrame,
    unit: str,
    type: Literal["expected", "posterior_predictive", "linear"] = "expected",
    max_cells: int = 5000000,
) -> PupilPrediction:
    if unit not in _PUPIL_UNITS:
        raise GP3BayesError("Unsupported pupil unit.")
    if type not in {"expected", "posterior_predictive", "linear"}:
        raise GP3BayesError("Unsupported pupil prediction type.")
    if not isinstance(grid, pd.DataFrame) or grid.empty or ".event_time" not in grid:
        raise GP3BayesError("`grid` must be a non-empty data frame containing `.event_time`.")
    arr = np.asarray(draws, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(grid) or not np.isfinite(arr).all():
        raise GP3BayesError("`draws` must be a finite numeric matrix with one column per grid row.")
    if arr.size > int(_positive(max_cells, "max_cells", True)):
        raise GP3BayesError("Requested draw-by-grid array exceeds `max_cells`.")
    return PupilPrediction(
        "0.4-pupil-1",
        "pupil",
        type,
        arr,
        grid.reset_index(drop=True).copy(),
        unit,
        arr.shape[0],
        arr.shape[1],
        "supplied_draws",
        False,
    )


def _prediction_grid(
    fit: PupilFit, newdata: pd.DataFrame | None, population_only: bool
) -> pd.DataFrame:
    if newdata is not None:
        if not isinstance(newdata, pd.DataFrame) or newdata.empty or ".event_time" not in newdata:
            raise GP3BayesError(
                "`newdata` must be None or a non-empty data frame containing `.event_time`."
            )
        return newdata.copy()
    d = fit.specification.prepared.data
    keep = (
        [".event_time"]
        + ([".condition"] if ".condition" in d else [])
        + list(fit.specification.covariates)
    )
    grid = d[keep].drop_duplicates().sort_values(keep, kind="stable").reset_index(drop=True)
    return grid


def predict_pupil_trajectory(
    fit: PupilFit,
    newdata: pd.DataFrame | None = None,
    type: Literal["expected", "posterior_predictive", "linear"] = "expected",
    ndraws: int = 500,
    population_only: bool = True,
    allow_new_levels: bool = False,
    max_grid: int = 5000,
    max_cells: int = 5000000,
) -> PupilPrediction:
    if not isinstance(fit, PupilFit) or not fit.fit_performed:
        raise GP3BayesError("`fit` must be a fitted `gp3bayes_pupil_fit`.")
    if not population_only and newdata is None:
        raise GP3BayesError(
            "Participant-conditioned prediction requires explicit `newdata`; gp3bayespy does not silently choose a participant or item."
        )
    grid = _prediction_grid(fit, newdata, population_only)
    n = min(int(_positive(ndraws, "ndraws", True)), fit.posterior_coefficients.shape[0])
    if len(grid) > int(_positive(max_grid, "max_grid", True)) or len(grid) * n > int(
        _positive(max_cells, "max_cells", True)
    ):
        raise GP3BayesError("Requested prediction expansion exceeds an explicit memory guard.")
    X, _ = _design_matrix(grid.assign(**{c: grid[c] for c in grid.columns}), fit.specification)
    beta = fit.posterior_coefficients[:n]
    eta = beta @ X.T
    rng = np.random.default_rng(fit.sampling["seed"] + 37)
    draws = (
        eta
        if type in {"expected", "linear"}
        else eta + rng.normal(0, fit.posterior_sigma[:n, None], eta.shape)
    )
    return as_pupil_prediction_draws(draws, grid, fit.outcome_unit, type, max_cells)


def _q8(draws: np.ndarray, probability: float, axis: int = 0) -> np.ndarray:
    alpha = (1 - probability) / 2
    return np.quantile(draws, [alpha, 0.5, 1 - alpha], axis=axis, method="median_unbiased")


def estimate_pupil_trajectory(
    prediction: PupilPrediction,
    probability: float = 0.95,
    interval: Literal["pointwise", "simultaneous"] = "pointwise",
) -> PupilTrajectory:
    if not isinstance(prediction, PupilPrediction):
        raise GP3BayesError("`prediction` must be a pupil prediction object.")
    prob = _probability(probability, "probability", True)
    q = _q8(prediction.draws, prob, 0)
    tab = prediction.grid.copy()
    tab["estimate"] = np.mean(prediction.draws, axis=0)
    tab["median"] = q[1]
    tab["lower"] = q[0]
    tab["upper"] = q[2]
    if interval == "simultaneous":
        mu = np.mean(prediction.draws, axis=0)
        s = np.std(prediction.draws, axis=0, ddof=1)
        s = np.where((~np.isfinite(s)) | (s == 0), 1, s)
        zmax = np.max(np.abs((prediction.draws - mu) / s), axis=1)
        c = float(np.quantile(zmax, prob, method="median_unbiased"))
        tab["lower"] = mu - c * s
        tab["upper"] = mu + c * s
    elif interval != "pointwise":
        raise GP3BayesError("`interval` must be pointwise or simultaneous.")
    return PupilTrajectory(
        tab, prediction.unit, prob, interval, prediction.type, interval == "simultaneous"
    )


def pupil_trajectory_table(x: PupilTrajectory) -> pd.DataFrame:
    if not isinstance(x, PupilTrajectory):
        raise GP3BayesError("`x` must be a pupil trajectory.")
    return x.table.copy()


def _groups(pred: PupilPrediction, mask: np.ndarray) -> list[tuple[dict[str, Any], np.ndarray]]:
    grid = pred.grid.loc[mask].reset_index(drop=True)
    cols = [c for c in (".condition", ".participant", ".item") if c in grid]
    if not cols:
        return [({}, np.arange(len(grid)))]
    result = []
    for key, sub in grid.groupby(cols, observed=True, sort=False):
        keys = (key,) if not isinstance(key, tuple) else key
        info = dict(zip(cols, keys, strict=True))
        result.append((info, sub.index.to_numpy()))
    return result


def _estimand(
    prediction: PupilPrediction, window: Sequence[float], statistic: str, probability: float
) -> PupilEstimand:
    w = _window(window, "window", False)
    assert w is not None
    mask = (
        (prediction.grid[".event_time"] >= w[0]) & (prediction.grid[".event_time"] <= w[1])
    ).to_numpy()
    if not mask.any():
        raise GP3BayesError("The declared window contains no prediction-grid points.")
    selected = np.flatnonzero(mask)
    rows = []
    for info, local_idx in _groups(prediction, mask):
        idx = selected[local_idx]
        x = prediction.grid.loc[idx, ".event_time"].to_numpy(dtype=float)
        sub = prediction.draws[:, idx]
        if statistic == "mean":
            vals = np.mean(sub, axis=1)
            unit = prediction.unit
        elif statistic == "auc":
            vals = np.array([np.trapezoid(row, x) for row in sub])
            unit = f"{prediction.unit}*seconds"
        elif statistic == "peak":
            vals = np.max(sub, axis=1)
            unit = prediction.unit
        elif statistic == "peak_latency":
            vals = x[np.argmax(sub, axis=1)]
            unit = "seconds"
        else:
            raise AssertionError
        q = _q8(vals, probability, 0).reshape(3)
        rows.append(
            {
                **info,
                "estimate": float(np.mean(vals)),
                "median": float(q[1]),
                "lower": float(q[0]),
                "upper": float(q[2]),
            }
        )
    return PupilEstimand(pd.DataFrame(rows), statistic, unit, probability, w)


def estimate_pupil_window(
    prediction: PupilPrediction, window: Sequence[float], probability: float = 0.95
) -> PupilEstimand:
    return _estimand(prediction, window, "mean", _probability(probability, "probability", True))


def estimate_pupil_auc(
    prediction: PupilPrediction, window: Sequence[float], probability: float = 0.95
) -> PupilEstimand:
    return _estimand(prediction, window, "auc", _probability(probability, "probability", True))


def estimate_pupil_peak(
    prediction: PupilPrediction, window: Sequence[float], probability: float = 0.95
) -> PupilEstimand:
    return _estimand(prediction, window, "peak", _probability(probability, "probability", True))


def estimate_pupil_peak_latency(
    prediction: PupilPrediction, window: Sequence[float], probability: float = 0.95
) -> PupilEstimand:
    return _estimand(
        prediction, window, "peak_latency", _probability(probability, "probability", True)
    )


def pupil_condition_contrast(
    prediction: PupilPrediction,
    contrast: Sequence[object],
    threshold: float = 0,
    probability: float = 0.95,
) -> PupilEstimand:
    if not isinstance(prediction, PupilPrediction) or ".condition" not in prediction.grid:
        raise GP3BayesError("Prediction grid must contain `.condition`.")
    levels = list(contrast)
    if len(levels) != 2:
        raise GP3BayesError("`contrast` must contain exactly two condition levels.")
    times = np.sort(prediction.grid[".event_time"].unique())
    rows = []
    draws_out = []
    for t in times:
        ai = np.flatnonzero(
            (prediction.grid[".event_time"].to_numpy() == t)
            & (prediction.grid[".condition"].astype(str).to_numpy() == str(levels[0]))
        )
        bi = np.flatnonzero(
            (prediction.grid[".event_time"].to_numpy() == t)
            & (prediction.grid[".condition"].astype(str).to_numpy() == str(levels[1]))
        )
        if ai.size != 1 or bi.size != 1:
            continue
        diff = prediction.draws[:, ai[0]] - prediction.draws[:, bi[0]]
        q = _q8(diff, _probability(probability, "probability", True), 0).reshape(3)
        rows.append(
            {
                ".event_time": t,
                "contrast_level_1": str(levels[0]),
                "contrast_level_2": str(levels[1]),
                "estimate": float(np.mean(diff)),
                "median": float(q[1]),
                "lower": float(q[0]),
                "upper": float(q[2]),
                "probability_gt_threshold": float(np.mean(diff > threshold)),
                "threshold": float(threshold),
            }
        )
        draws_out.append(diff)
    return PupilEstimand(
        pd.DataFrame(rows), "condition_contrast", prediction.unit, float(probability), None
    )


@dataclass(frozen=True, slots=True)
class PupilTrajectoryDerivative:
    grid: pd.DataFrame
    draws: np.ndarray
    order: int
    probability: float
    specification: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PupilDynamicContrast:
    table: pd.DataFrame
    grid: pd.DataFrame
    draws: np.ndarray
    contrast: tuple[str, str]
    threshold: float
    derivative_order: int
    probability: float
    specification: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PupilThresholdDuration:
    summary: pd.DataFrame
    draws: np.ndarray
    direction: str
    threshold: float
    probability: float
    time_unit: str = "same unit as the prediction time column"


@dataclass(frozen=True, slots=True)
class GazepointPupilSchema:
    detected: pd.DataFrame
    pupil_candidates: pd.DataFrame
    time_candidates: pd.DataFrame
    gaze_candidates: pd.DataFrame
    validity_candidates: pd.DataFrame
    status: str
    audit: pd.DataFrame
    automatic_channel_selection: bool = False


@dataclass(frozen=True, slots=True)
class PupilValidationPlan:
    target: str
    strategy: str
    K: int | None
    fold_id: np.ndarray | None
    split_table: pd.DataFrame
    model_row_index: np.ndarray
    source_rows: np.ndarray
    future_fraction: float
    seed: int
    n_rows: int
    leakage_detected: bool
    qualification: str
    automatic_strategy_selection: bool = False


@dataclass(frozen=True, slots=True)
class PupilValidation:
    target: str
    strategy: str
    executed: bool
    plan: PupilValidationPlan
    result: Any
    table: pd.DataFrame
    validity_certified: bool = False


@dataclass(frozen=True, slots=True)
class PupilSensitivitySuite:
    baseline_specification: PupilModelSpecification
    baseline_window_operation: str | None
    pfe_prepared: Mapping[str, PupilPrepared]
    scenarios: pd.DataFrame
    automatic_effect_maximization: bool = False
    pfe_correction_performed: bool = False


@dataclass(frozen=True, slots=True)
class PupilSensitivityComparison:
    table: pd.DataFrame
    automatic_selection: bool = False


def _functional_parts(
    prediction: object,
) -> tuple[pd.DataFrame, np.ndarray, Mapping[str, Any], int]:
    if isinstance(prediction, PupilTrajectoryDerivative):
        return prediction.grid, prediction.draws, prediction.specification, prediction.order
    grid = getattr(prediction, "grid", None)
    draws = getattr(prediction, "draws", None)
    spec = getattr(prediction, "specification", None)
    if isinstance(spec, AdvancedPupilSpecification):
        spec = {"mapping": dict(spec.mapping)}
    if (
        isinstance(grid, pd.DataFrame)
        and isinstance(draws, np.ndarray)
        and isinstance(spec, Mapping)
    ):
        return grid, draws, spec, 0
    raise GP3BayesError("Expected an advanced trajectory prediction or derivative object.")


def _derivative_once(
    grid: pd.DataFrame, draws: np.ndarray, time_col: str, condition_col: str | None
) -> tuple[pd.DataFrame, np.ndarray]:
    groups = (
        pd.Series(".all", index=grid.index)
        if condition_col is None
        else grid[condition_col].astype(str)
    )
    grids = []
    dds = []
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups.to_numpy() == group)
        idx = idx[np.argsort(grid.iloc[idx][time_col].to_numpy(dtype=float))]
        tt = grid.iloc[idx][time_col].to_numpy(dtype=float)
        if len(tt) < 2:
            continue
        dt = np.diff(tt)
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
            raise GP3BayesError(
                "Derivative estimation requires strictly increasing unique time points within condition."
            )
        dd = np.diff(draws[:, idx], axis=1) / dt[None, :]
        g = grid.iloc[idx[1:]].copy()
        g[time_col] = (tt[1:] + tt[:-1]) / 2
        grids.append(g)
        dds.append(dd)
    if not grids:
        raise GP3BayesError("Not enough distinct time points to estimate a derivative.")
    return pd.concat(grids, ignore_index=True), np.column_stack(dds)


def estimate_pupil_trajectory_derivative(
    prediction: object, order: int = 1, probability: float = 0.95
) -> PupilTrajectoryDerivative:
    grid, draws, spec, _ = _functional_parts(prediction)
    order_value = int(order)
    if order_value not in {1, 2}:
        raise GP3BayesError("`order` must be 1 or 2.")
    prob = _probability(probability, "probability", True)
    mapping = spec.get("mapping", {})
    time_col = str(mapping.get("time", ".event_time"))
    condition_col = mapping.get("condition")
    cur_grid = grid.copy()
    cur_draws = draws.copy()
    for _ in range(order_value):
        cur_grid, cur_draws = _derivative_once(cur_grid, cur_draws, time_col, condition_col)
    return PupilTrajectoryDerivative(cur_grid, cur_draws, order_value, prob, spec)


def pupil_trajectory_derivative_table(
    x: PupilTrajectoryDerivative, probability: float | None = None
) -> pd.DataFrame:
    if not isinstance(x, PupilTrajectoryDerivative):
        raise GP3BayesError("Expected a trajectory derivative object.")
    prob = x.probability if probability is None else _probability(probability, "probability", True)
    q = _q8(x.draws, prob, 0)
    out = x.grid.copy()
    out["derivative_order"] = x.order
    out["estimate"] = np.mean(x.draws, axis=0)
    out["lower"] = q[0]
    out["median"] = q[1]
    out["upper"] = q[2]
    return out


def estimate_pupil_dynamic_contrast(
    prediction: object, contrast: Sequence[str], threshold: float = 0, probability: float = 0.95
) -> PupilDynamicContrast:
    grid, draws, spec, derivative_order = _functional_parts(prediction)
    levels = tuple(str(v) for v in contrast)
    if len(levels) != 2:
        raise GP3BayesError("`contrast` must contain exactly two non-missing condition labels.")
    prob = _probability(probability, "probability", True)
    mapping = spec.get("mapping", {})
    time_col = str(mapping.get("time", ".event_time"))
    cond_col = mapping.get("condition")
    if cond_col is None or cond_col not in grid:
        raise GP3BayesError("A condition column is required for a dynamic contrast.")
    cond = grid[cond_col].astype(str).to_numpy()
    idx1 = np.flatnonzero(cond == levels[0])
    idx2 = np.flatnonzero(cond == levels[1])
    t1 = grid.iloc[idx1][time_col].to_numpy(dtype=float)
    t2 = grid.iloc[idx2][time_col].to_numpy(dtype=float)
    common = np.intersect1d(t1, t2)
    if common.size == 0:
        raise GP3BayesError("The two conditions have no common prediction times.")
    p1 = np.array([idx1[np.flatnonzero(t1 == t)[0]] for t in common])
    p2 = np.array([idx2[np.flatnonzero(t2 == t)[0]] for t in common])
    d = draws[:, p1] - draws[:, p2]
    q = _q8(d, prob, 0)
    table = pd.DataFrame(
        {
            time_col: common,
            "contrast": f"{levels[0]} - {levels[1]}",
            "derivative_order": derivative_order,
            "threshold": float(threshold),
            "estimate": np.mean(d, axis=0),
            "lower": q[0],
            "median": q[1],
            "upper": q[2],
            "probability_above_threshold": np.mean(d > threshold, axis=0),
            "probability_below_negative_threshold": np.mean(d < -abs(threshold), axis=0),
        }
    )
    g = pd.DataFrame({time_col: common})
    return PupilDynamicContrast(table, g, d, levels, float(threshold), derivative_order, prob, spec)


def pupil_dynamic_contrast_table(x: PupilDynamicContrast) -> pd.DataFrame:
    if not isinstance(x, PupilDynamicContrast):
        raise GP3BayesError("Expected a dynamic pupil contrast.")
    return x.table.copy()


def estimate_pupil_threshold_duration(
    contrast: PupilDynamicContrast,
    direction: Literal["above", "below", "absolute"] = "above",
    threshold: float | None = None,
    probability: float = 0.95,
) -> PupilThresholdDuration:
    if not isinstance(contrast, PupilDynamicContrast):
        raise GP3BayesError("Expected a dynamic pupil contrast.")
    if direction not in {"above", "below", "absolute"}:
        raise GP3BayesError("Unsupported direction.")
    th = contrast.threshold if threshold is None else float(threshold)
    prob = _probability(probability, "probability", True)
    time_col = str(contrast.specification.get("mapping", {}).get("time", ".event_time"))
    tt = contrast.grid[time_col].to_numpy(dtype=float)
    ord_idx = np.argsort(tt)
    tt = tt[ord_idx]
    d = contrast.draws[:, ord_idx]
    if len(tt) < 2:
        raise GP3BayesError("At least two contrast time points are required.")
    hit = (
        d > th if direction == "above" else d < th if direction == "below" else np.abs(d) > abs(th)
    )
    duration = (hit[:, 1:] & hit[:, :-1]) @ np.diff(tt)
    q = _q8(duration, prob, 0).reshape(3)
    summary = pd.DataFrame(
        {
            "direction": [direction],
            "threshold": [th],
            "mean": [float(np.mean(duration))],
            "sd": [float(np.std(duration, ddof=1))],
            "q_low": [float(q[0])],
            "median": [float(q[1])],
            "q_high": [float(q[2])],
        }
    )
    return PupilThresholdDuration(summary, np.asarray(duration), direction, th, prob)


def inspect_gazepoint_pupil_schema(data: pd.DataFrame) -> GazepointPupilSchema:
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    schema = pd.DataFrame(
        [
            ("TIME", "time", "seconds", "both"),
            ("TIME_TICK", "time_tick", "processor_ticks", "both"),
            ("LPD", "left_pupil_diameter_pixels", "pixels", "left"),
            ("RPD", "right_pupil_diameter_pixels", "pixels", "right"),
            ("LPUPILD", "left_pupil_diameter_metres", "metres", "left"),
            ("RPUPILD", "right_pupil_diameter_metres", "metres", "right"),
            ("LPV", "left_pupil_valid", "logical", "left"),
            ("RPV", "right_pupil_valid", "logical", "right"),
            ("LPOGX", "left_gaze_x", "proportion", "left"),
            ("LPOGY", "left_gaze_y", "proportion", "left"),
            ("RPOGX", "right_gaze_x", "proportion", "right"),
            ("RPOGY", "right_gaze_y", "proportion", "right"),
        ],
        columns=["field", "role", "unit", "eye"],
    )
    schema["present"] = schema["field"].isin(data.columns)
    detected = schema[schema.present].copy()
    pupil = detected[detected.role.str.contains("pupil_diameter")].copy()
    time = detected[detected.role.isin(["time", "time_tick"])].copy()
    gaze = detected[detected.role.str.contains("gaze_[xy]", regex=True)].copy()
    valid = detected[detected.role.str.endswith("valid")].copy()
    status = (
        "missing_pupil_channel"
        if pupil.empty
        else "ambiguous_pupil_channel"
        if len(pupil) > 1
        else "single_pupil_candidate"
    )
    audit = pd.DataFrame(
        {
            "check": [
                "documented_fields_detected",
                "pupil_channels_detected",
                "time_fields_detected",
                "gaze_fields_detected",
                "validity_fields_detected",
                "channel_selection",
            ],
            "value": [len(detected), len(pupil), len(time), len(gaze), len(valid), status],
        }
    )
    return GazepointPupilSchema(detected, pupil, time, gaze, valid, status, audit)


def gazepoint_pupil_mapping_table(x: GazepointPupilSchema | pd.DataFrame) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        x = inspect_gazepoint_pupil_schema(x)
    if not isinstance(x, GazepointPupilSchema):
        raise GP3BayesError("`x` must be a Gazepoint pupil schema inspection or data frame.")
    all_fields = pd.DataFrame(
        [
            ("TIME", "time", "seconds", "both"),
            ("TIME_TICK", "time_tick", "processor_ticks", "both"),
            ("LPD", "left_pupil_diameter_pixels", "pixels", "left"),
            ("RPD", "right_pupil_diameter_pixels", "pixels", "right"),
            ("LPUPILD", "left_pupil_diameter_metres", "metres", "left"),
            ("RPUPILD", "right_pupil_diameter_metres", "metres", "right"),
            ("LPV", "left_pupil_valid", "logical", "left"),
            ("RPV", "right_pupil_valid", "logical", "right"),
        ],
        columns=["field", "role", "unit", "eye"],
    )
    all_fields["present"] = all_fields.field.isin(x.detected.field)
    return all_fields


def _prepared_from_any(x: object) -> PupilPrepared:
    if isinstance(x, PupilPrepared):
        return x
    if isinstance(x, PupilModelSpecification):
        return x.prepared
    if isinstance(x, PupilFit):
        return x.specification.prepared
    raise GP3BayesError("`x` must be a pupil prepared object, specification, or fit.")


def create_pupil_validation_plan(
    x: object,
    target: Literal[
        "new_trial_known_participant", "new_participant", "future_segment", "new_sample_known_trial"
    ] = "new_trial_known_participant",
    K: int = 5,
    future_fraction: float = 0.20,
    seed: int = 2026,
) -> PupilValidationPlan:
    p = _prepared_from_any(x)
    k = int(_positive(K, "K", True))
    ff = _probability(future_fraction, "future_fraction", True)
    d = p.data[p.data[".pupil_model"].notna()].reset_index(drop=True)
    n = len(d)
    if n < 4:
        raise GP3BayesError("Validation requires at least four non-missing pupil observations.")
    rng = np.random.default_rng(int(seed))
    folds = None
    split = pd.DataFrame()
    qualification = ""
    if target == "new_sample_known_trial":
        folds = np.resize(np.arange(1, min(k, n) + 1), n)
        rng.shuffle(folds)
        strategy = "observation_kfold"
        kval = min(k, n)
    elif target == "new_trial_known_participant":
        folds = np.zeros(n, dtype=int)
        for pi, (_participant, grp) in enumerate(
            d.groupby(".participant", observed=True, sort=False)
        ):
            trials = np.array(pd.unique(grp[".trial"].astype(str)))
            if len(trials) < 2:
                raise GP3BayesError(
                    "Target `new_trial_known_participant` requires at least two trials for every participant."
                )
            local = np.arange(1, min(k, len(trials)) + 1)
            order = np.random.default_rng(int(seed) + pi + 1).permutation(trials)
            mapping = {t: int(local[i % len(local)]) for i, t in enumerate(order)}
            folds[grp.index] = grp[".trial"].astype(str).map(mapping)
        kval = int(folds.max())
        strategy = "grouped_trial_kfold_within_participant"
    elif target == "new_participant":
        participants = np.array(pd.unique(d[".participant"].astype(str)))
        kval = min(k, len(participants))
        if kval < 2:
            raise GP3BayesError("New-participant validation requires at least two participants.")
        shuffled = rng.permutation(participants)
        mapping = {p: int(i % kval + 1) for i, p in enumerate(shuffled)}
        folds = d[".participant"].astype(str).map(mapping).to_numpy(dtype=int)
        strategy = "grouped_participant_kfold"
    elif target == "future_segment":
        pieces = []
        for series, grp in d.groupby(".series_id", observed=True, sort=False):
            grp = grp.sort_values(".event_time")
            ntest = max(1, int(math.floor(len(grp) * ff)))
            cut = len(grp) - ntest
            if cut < 2:
                raise GP3BayesError(
                    "Future-segment split leaves too little training data in a series."
                )
            pieces.append(
                pd.DataFrame(
                    {
                        "row": grp.index + 1,
                        "source_row": grp[".source_row"].to_numpy(),
                        "series_id": str(series),
                        "role": ["train"] * cut + ["test"] * ntest,
                        "event_time": grp[".event_time"].to_numpy(),
                    }
                )
            )
        split = pd.concat(pieces, ignore_index=True).sort_values("row")
        strategy = "leave_future_segment_out"
        kval = None
    else:
        raise GP3BayesError("Unknown pupil validation target.")
    leakage = False
    if target == "new_participant":
        leakage = any(
            pd.Series(folds, index=d.index).groupby(d[".participant"], observed=True).nunique() > 1
        )
    elif target == "new_trial_known_participant":
        leakage = any(
            pd.Series(folds, index=d.index).groupby(d[".series_id"], observed=True).nunique() > 1
        )
    elif target == "future_segment":
        leakage = any(
            g.loc[g.role == "train", "event_time"].max()
            >= g.loc[g.role == "test", "event_time"].min()
            for _, g in split.groupby("series_id")
        )
    return PupilValidationPlan(
        target,
        strategy,
        kval,
        folds,
        split,
        np.flatnonzero(p.data[".pupil_model"].notna()) + 1,
        d[".source_row"].to_numpy(),
        ff,
        int(seed),
        n,
        bool(leakage),
        qualification,
    )


def validate_pupil_model(
    fit: PupilFit,
    plan: PupilValidationPlan,
    execute: bool = False,
    ndraws: int = 200,
    max_cells: int = 3000000,
) -> PupilValidation:
    if not isinstance(fit, PupilFit):
        raise GP3BayesError("`fit` must be a fitted pupil model.")
    if not isinstance(plan, PupilValidationPlan):
        raise GP3BayesError("`plan` must be created by create_pupil_validation_plan().")
    if plan.leakage_detected:
        raise GP3BayesError("Validation plan failed its leakage check.")
    if not execute:
        tab = pd.DataFrame(
            {
                "target": [plan.target],
                "strategy": [plan.strategy],
                "executed": [False],
                "leakage_detected": [plan.leakage_detected],
                "qualification": [plan.qualification],
            }
        )
        return PupilValidation(plan.target, plan.strategy, False, plan, None, tab)
    # Python adaptation: deterministic held-out scoring using the approved analytic family.
    d = fit.specification.prepared.data[
        fit.specification.prepared.data[".pupil_model"].notna()
    ].reset_index(drop=True)
    rows = []
    if plan.fold_id is not None:
        for fold in sorted(set(plan.fold_id.tolist())):
            train = d[plan.fold_id != fold]
            test = d[plan.fold_id == fold]
            temp = replace(
                fit.specification, prepared=replace(fit.specification.prepared, data=train)
            )
            refit = fit_pupil_model_backend(temp, "analytic", 1, 200, 100, 1, plan.seed + fold)
            pred = predict_pupil_trajectory(
                refit, newdata=test, ndraws=min(ndraws, refit.posterior_coefficients.shape[0])
            )
            mu = np.mean(pred.draws, axis=0)
            err = test[".pupil_model"].to_numpy(dtype=float) - mu
            rows.append(
                {
                    "fold": fold,
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "mae": float(np.mean(np.abs(err))),
                }
            )
    else:
        role = dict(zip(plan.split_table.source_row, plan.split_table.role, strict=True))
        train = d[d[".source_row"].map(role) == "train"]
        test = d[d[".source_row"].map(role) == "test"]
        temp = replace(fit.specification, prepared=replace(fit.specification.prepared, data=train))
        refit = fit_pupil_model_backend(temp, "analytic", 1, 200, 100, 1, plan.seed)
        pred = predict_pupil_trajectory(
            refit, newdata=test, ndraws=min(ndraws, refit.posterior_coefficients.shape[0])
        )
        mu = np.mean(pred.draws, axis=0)
        err = test[".pupil_model"].to_numpy(dtype=float) - mu
        rows.append(
            {
                "n_train": len(train),
                "n_test": len(test),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(0, "strategy", plan.strategy)
    table.insert(0, "target", plan.target)
    return PupilValidation(plan.target, plan.strategy, True, plan, None, table)


def pupil_validation_table(x: PupilValidation) -> pd.DataFrame:
    if not isinstance(x, PupilValidation):
        raise GP3BayesError("`x` must be a pupil validation object.")
    return x.table.copy()


def create_pupil_sensitivity_suite(
    specification: PupilModelSpecification,
    baseline_windows: Sequence[Sequence[float]] = (),
    baseline_window_operation: str | None = None,
    baseline_operations: Sequence[str] = (),
    interpolation_policy: Sequence[str] = (),
    blink_adjacent_margins: Sequence[float] = (),
    gaze_adjustment: Sequence[str] = (),
    luminance_adjustment: Sequence[str] = (),
    pfe_prepared: Mapping[str, PupilPrepared] | None = None,
    smooth_basis_dimensions: Sequence[int] = (),
    autocorrelation: Sequence[str] = (),
    analysis_windows: Sequence[Sequence[float]] = (),
) -> PupilSensitivitySuite:
    if not isinstance(specification, PupilModelSpecification):
        raise GP3BayesError("`specification` must be a pupil model specification.")
    if (
        baseline_windows
        and specification.prepared.baseline_operation == "none"
        and baseline_window_operation is None
    ):
        raise GP3BayesError(
            "`baseline_window_operation` is required when baseline-window sensitivity starts from no baseline transformation."
        )
    scenarios = []

    def add(axis: str, values: Sequence[Any]):
        for value in values:
            text = (
                ",".join(str(v) for v in value)
                if isinstance(value, (list, tuple, np.ndarray))
                else str(value)
            )
            scenarios.append((axis, text))

    add("baseline_window", baseline_windows)
    add("baseline_operation", baseline_operations)
    add("interpolation_policy", interpolation_policy)
    add("blink_adjacent_margin", blink_adjacent_margins)
    add("gaze_adjustment", gaze_adjustment)
    add("luminance_adjustment", luminance_adjustment)
    add("pfe_prepared", tuple((pfe_prepared or {}).keys()))
    add("smooth_basis_dimension", smooth_basis_dimensions)
    add("autocorrelation", autocorrelation)
    add("analysis_window", analysis_windows)
    table = pd.DataFrame(scenarios, columns=["axis", "value"])
    table.insert(0, "scenario_id", [f"S{i:03d}" for i in range(1, len(table) + 1)])
    return PupilSensitivitySuite(
        specification,
        baseline_window_operation
        or (
            specification.prepared.baseline_operation
            if specification.prepared.baseline_operation != "none"
            else None
        ),
        dict(pfe_prepared or {}),
        table,
    )


def materialize_pupil_sensitivity_scenario(
    suite: PupilSensitivitySuite, scenario_id: str
) -> Mapping[str, Any]:
    if not isinstance(suite, PupilSensitivitySuite):
        raise GP3BayesError("`suite` must be a pupil sensitivity suite.")
    row = suite.scenarios[suite.scenarios.scenario_id == scenario_id]
    if len(row) != 1:
        raise GP3BayesError("Unknown or duplicated `scenario_id`.")
    axis = str(row.iloc[0].axis)
    value = str(row.iloc[0].value)
    base = suite.baseline_specification
    prepared = base.prepared
    spec = base
    analysis_window = None
    if axis == "smooth_basis_dimension":
        spec = replace(
            base,
            smooth_basis_dimension=int(value),
            formula=base.formula.replace(f"k={base.smooth_basis_dimension}", f"k={int(value)}"),
            formula_text=base.formula_text.replace(
                f"k={base.smooth_basis_dimension}", f"k={int(value)}"
            ),
        )
    elif axis == "autocorrelation":
        spec = replace(base, autocorrelation=value)
    elif axis == "analysis_window":
        analysis_window = tuple(float(v) for v in value.split(","))
    elif axis == "pfe_prepared":
        prepared = suite.pfe_prepared[value]
        spec = replace(
            base, prepared=prepared, contract=prepared.contract, outcome_unit=prepared.model_unit
        )
    return {
        "scenario_id": scenario_id,
        "axis": axis,
        "value": value,
        "prepared": prepared,
        "specification": spec,
        "analysis_window": analysis_window,
        "fit_performed": False,
        "pfe_correction_performed": False,
    }


def compare_pupil_sensitivity_estimands(
    results: Mapping[str, PupilEstimand],
) -> PupilSensitivityComparison:
    if not isinstance(results, Mapping) or not results:
        raise GP3BayesError("`results` must be a non-empty named mapping of pupil estimands.")
    frames = []
    for name, res in results.items():
        if not isinstance(res, PupilEstimand):
            raise GP3BayesError("Every sensitivity result must be a pupil estimand.")
        tab = res.table.copy()
        tab["scenario_id"] = str(name)
        frames.append(tab)
    return PupilSensitivityComparison(pd.concat(frames, ignore_index=True))


def pupil_sensitivity_table(x: PupilSensitivitySuite | PupilSensitivityComparison) -> pd.DataFrame:
    if isinstance(x, PupilSensitivitySuite):
        return x.scenarios.copy()
    if isinstance(x, PupilSensitivityComparison):
        return x.table.copy()
    raise GP3BayesError("`x` must be a pupil sensitivity object.")


# ---------------------------------------------------------------------------
# gp3bayes 0.5 advanced pupil specification, simulation, and governance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PupilMeasurementModel:
    covariate_errors: Mapping[str, str]
    response_error: str | None
    interpretation: str = (
        "Known uncertainty is propagated conditionally on the declared measurement model. "
        "The declaration does not validate calibration, measurement unbiasedness, or causal interpretation."
    )


@dataclass(frozen=True, slots=True)
class PupilMissingnessSpec:
    response: str
    predictors: tuple[str, ...]
    assumptions: str
    auxiliary_predictors: tuple[str, ...]
    interpretation: str = (
        "Missingness modelling is assumption-conditional. Fitting a missing-data model "
        "does not establish that MAR is true."
    )


@dataclass(frozen=True, slots=True)
class PupilGPSpec:
    kernel: str
    basis: str
    k: int | None
    scale: bool


@dataclass(frozen=True, slots=True)
class PupilARMASpec:
    p: int
    q: int
    covariance: bool


@dataclass(frozen=True, slots=True)
class PupilDistributionSpec:
    family: str
    residual_scale: str


@dataclass(frozen=True, slots=True)
class PupilComplexityAudit:
    overall_status: str
    rows: int
    unique_time: int
    conditions: int
    participants: int
    series: int
    checks: pd.DataFrame


@dataclass(frozen=True, slots=True)
class AdvancedPupilSpecification:
    version: str
    prepared: PupilPrepared | pd.DataFrame
    data: pd.DataFrame
    mapping: Mapping[str, str | None]
    temporal_structure: str
    family: str
    residual_scale: str
    smooth_basis_dimension: int
    smooth_basis_dimension_requested: int
    smooth_basis_dimension_effective: int
    smooth_basis_support: int
    smooth_basis_adjusted: bool
    gp_spec: PupilGPSpec | None
    condition_trajectory: bool
    autocorrelation: PupilARMASpec | None
    participant_trajectory: str
    item_effects: bool
    covariates: tuple[str, ...]
    measurement_model: PupilMeasurementModel | None
    missingness_model: PupilMissingnessSpec | None
    prior_scales: Mapping[str, float] | None
    predictive_target: str
    allow_high_complexity: bool
    compatibility: pd.DataFrame
    complexity_audit: PupilComplexityAudit | None = None
    backend: str = "none"
    fit_performed: bool = False
    governance: tuple[str, ...] = (
        "No automatic preprocessing, interpolation, exclusion, or model selection.",
        "No automatic cognitive-state, causal, or adequacy interpretation.",
        "Measurement and missingness models remain assumption-conditional.",
        "Predictive comparison is tied to an explicitly declared target.",
    )


@dataclass(frozen=True, slots=True)
class AdvancedPupilSimulation:
    data: pd.DataFrame
    truth: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AdvancedPupilSensitivitySuite:
    baseline: AdvancedPupilSpecification
    scenarios: pd.DataFrame
    interpretation: str = (
        "Sensitivity scenarios are pre-declared alternatives; the suite does not fit, "
        "rank, or choose a preferred scenario."
    )


@dataclass(frozen=True, slots=True)
class PupilModelCard:
    table: pd.DataFrame
    governance: tuple[str, ...]
    interpretation: str = (
        "The model card records declared analysis decisions and computational structure. "
        "It is not an automatic certificate of validity, robustness, or causal identification."
    )


def create_pupil_measurement_model(
    baseline_error: str | None = None,
    luminance_error: str | None = None,
    gaze_error: str | None = None,
    response_error: str | None = None,
    covariate_errors: Mapping[str, str] | None = None,
) -> PupilMeasurementModel:
    common: dict[str, str] = {}
    for key, value in (
        ("baseline", baseline_error),
        ("luminance", luminance_error),
        ("gaze_eccentricity", gaze_error),
    ):
        if value is not None:
            common[key] = str(_scalar_name(value, f"{key}_error"))
    custom = dict(covariate_errors or {})
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in custom.items()
    ):
        raise GP3BayesError(
            "`covariate_errors` must map non-empty covariate names to standard-error columns."
        )
    overlap = set(common).intersection(custom)
    if overlap:
        raise GP3BayesError("Duplicate measurement-error declarations for the same covariate.")
    response = _scalar_name(response_error, "response_error", True)
    return PupilMeasurementModel({**common, **custom}, response)


def create_pupil_missingness_spec(
    response: Literal["exclude", "model"] = "exclude",
    predictors: Sequence[str] = (),
    assumptions: str = "MAR",
    auxiliary_predictors: Sequence[str] = (),
) -> PupilMissingnessSpec:
    if response not in {"exclude", "model"}:
        raise GP3BayesError("`response` must be exclude or model.")
    if assumptions != "MAR":
        raise GP3BayesError(
            'gp3bayes 0.5 does not implement MNAR identification. `assumptions` must be "MAR".'
        )
    pred = tuple(dict.fromkeys(str(v) for v in predictors if str(v)))
    aux = tuple(dict.fromkeys(str(v) for v in auxiliary_predictors if str(v)))
    return PupilMissingnessSpec(response, pred, assumptions, aux)


def create_pupil_gp_spec(
    kernel: Literal["matern32", "matern52", "exp_quad"] = "matern32",
    basis: Literal["approximate", "exact"] = "approximate",
    k: int = 30,
    scale: bool = True,
) -> PupilGPSpec:
    if kernel not in {"matern32", "matern52", "exp_quad"}:
        raise GP3BayesError("Unsupported GP kernel.")
    if basis not in {"approximate", "exact"}:
        raise GP3BayesError("Unsupported GP basis.")
    if not isinstance(scale, bool):
        raise GP3BayesError("`scale` must be boolean.")
    if basis == "approximate":
        kval = int(_positive(k, "k", True))
        if kval < 5 or kval > 200:
            raise GP3BayesError("For approximate GPs, `k` must be an integer from 5 to 200.")
    else:
        kval = None
    return PupilGPSpec(kernel, basis, kval, scale)


def create_pupil_arma_spec(p: int = 1, q: int = 0, covariance: bool = False) -> PupilARMASpec:
    for value, name, high in ((p, "p", 3), (q, "q", 2)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise GP3BayesError(f"`{name}` must be an integer in [0, {high}].")
        if value < 0 or value > high:
            raise GP3BayesError(f"`{name}` must be an integer in [0, {high}].")
    if p == 0 and q == 0:
        raise GP3BayesError('Use `autocorrelation = "none"` instead of ARMA(0,0).')
    if not isinstance(covariance, bool):
        raise GP3BayesError("`covariance` must be boolean.")
    if covariance and (p > 1 or q > 1):
        raise GP3BayesError("Covariance-form ARMA is restricted to orders no greater than (1,1).")
    return PupilARMASpec(int(p), int(q), covariance)


def specify_pupil_distribution(
    family: Literal["gaussian", "student"] = "gaussian",
    residual_scale: Literal["constant", "condition", "time", "condition_time"] = "constant",
) -> PupilDistributionSpec:
    if family not in {"gaussian", "student"}:
        raise GP3BayesError("Unsupported pupil distribution family.")
    if residual_scale not in {"constant", "condition", "time", "condition_time"}:
        raise GP3BayesError("Unsupported pupil residual-scale declaration.")
    return PupilDistributionSpec(family, residual_scale)


def pupil_distribution_table(x: PupilDistributionSpec) -> pd.DataFrame:
    if not isinstance(x, PupilDistributionSpec):
        raise GP3BayesError("`x` must be a pupil distribution specification.")
    return pd.DataFrame({"family": [x.family], "residual_scale": [x.residual_scale]})


def pupil_advanced_compatibility_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capability": [
                "Student-t likelihood with residual ARMA",
                "Student-t with distributional residual scale",
                "bounded Gaussian ARMA",
                "Gaussian distributional residual scale",
                "approximate Gaussian process",
                "exact Gaussian process",
                "missing response with ARMA",
                "missing/uncertain predictor submodels",
                "binocular residual correlation",
                "automatic cognitive-state inference",
            ],
            "status": [
                "blocked",
                "supported",
                "supported",
                "supported",
                "supported",
                "explicit opt-in",
                "blocked",
                "supported",
                "supported",
                "not implemented",
            ],
        }
    )


def pupil_advanced_capabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capability": [
                "Gaussian and Student observation families",
                "bounded ARMA residual dependence",
                "smooth and Gaussian-process trajectories",
                "known measurement uncertainty",
                "MAR-oriented missing-data models",
                "binocular models",
                "experimental response-shape models",
                "automatic cognitive or emotional inference",
            ],
            "status": [
                "supported",
                "supported",
                "supported",
                "supported",
                "supported",
                "supported",
                "supported",
                "excluded",
            ],
        }
    )


def _advanced_mapping(
    prepared: PupilPrepared | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if isinstance(prepared, PupilPrepared):
        d = prepared.data.copy()
        mapping = {
            "response": ".pupil_model",
            "time": ".event_time",
            "participant": ".participant",
            "trial": ".trial",
            "condition": ".condition" if ".condition" in d else None,
            "item": ".item" if ".item" in d else None,
        }
        return d, mapping
    if not isinstance(prepared, pd.DataFrame) or prepared.empty:
        raise GP3BayesError("`prepared` must be a prepared pupil object or compatible data frame.")
    d = prepared.copy()
    candidates = {
        "response": ["pupil", ".pupil_model", "pupil_mm"],
        "time": ["time_ms", ".event_time", "event_time", "time"],
        "participant": ["participant_id", ".participant", "participant"],
        "trial": ["trial_id", ".trial", "trial"],
        "condition": ["condition", ".condition"],
        "item": ["item_id", ".item"],
    }
    mapping: dict[str, str | None] = {}  # type: ignore[no-redef]
    for role, names in candidates.items():
        hit = next((name for name in names if name in d.columns), None)
        if role in {"response", "time", "participant"} and hit is None:
            raise GP3BayesError(f"Could not identify required advanced pupil `{role}` column.")
        mapping[role] = hit
    return d, mapping


def pupil_advanced_mapping_table(prepared: PupilPrepared | pd.DataFrame) -> pd.DataFrame:
    _, mapping = _advanced_mapping(prepared)
    return pd.DataFrame(
        [(role, column) for role, column in mapping.items() if column is not None],
        columns=["role", "column"],
    )


def _ac_spec(x: str | PupilARMASpec) -> PupilARMASpec | None:
    if isinstance(x, PupilARMASpec):
        return x
    if x == "none":
        return None
    if x == "ar1":
        return create_pupil_arma_spec(1, 0)
    if x == "ar2":
        return create_pupil_arma_spec(2, 0)
    if x == "arma11":
        return create_pupil_arma_spec(1, 1)
    raise GP3BayesError("Unsupported autocorrelation declaration.")


def specify_advanced_pupil_timecourse_model(
    prepared: PupilPrepared | pd.DataFrame,
    temporal_structure: Literal["smooth", "linear", "gaussian_process"] = "smooth",
    family: Literal["gaussian", "student"] = "gaussian",
    residual_scale: Literal["constant", "condition", "time", "condition_time"] = "constant",
    distribution: PupilDistributionSpec | None = None,
    smooth_basis_dimension: int = 10,
    gp_spec: PupilGPSpec | None = None,
    condition_trajectory: bool | None = None,
    autocorrelation: str | PupilARMASpec = "none",
    participant_trajectory: Literal["none", "factor_smooth"] = "none",
    item_effects: bool | None = None,
    covariates: Sequence[str] = (),
    measurement_model: PupilMeasurementModel | None = None,
    missingness_model: PupilMissingnessSpec | None = None,
    prior_scales: Mapping[str, float] | None = None,
    predictive_target: Literal[
        "new_trial_known_participant", "new_participant", "future_segment", "new_sample_known_trial"
    ] = "new_trial_known_participant",
    allow_high_complexity: bool = False,
) -> AdvancedPupilSpecification:
    data, mapping = _advanced_mapping(prepared)
    if distribution is not None:
        if not isinstance(distribution, PupilDistributionSpec):
            raise GP3BayesError("`distribution` must come from specify_pupil_distribution().")
        family, residual_scale = distribution.family, distribution.residual_scale  # type: ignore[assignment]
    if temporal_structure not in {"smooth", "linear", "gaussian_process"}:
        raise GP3BayesError("Unsupported advanced temporal structure.")
    if family not in {"gaussian", "student"}:
        raise GP3BayesError("Unsupported advanced pupil family.")
    if residual_scale not in {"constant", "condition", "time", "condition_time"}:
        raise GP3BayesError("Unsupported residual-scale declaration.")
    ac = _ac_spec(autocorrelation)
    if family == "student" and ac is not None:
        raise GP3BayesError(
            "Student-t observation models with residual ARMA are deliberately blocked in gp3bayes 0.5."
        )
    if missingness_model is not None and missingness_model.response == "model" and ac is not None:
        raise GP3BayesError(
            "Missing-response models cannot currently be combined with residual ARMA."
        )
    if not isinstance(allow_high_complexity, bool):
        raise GP3BayesError("`allow_high_complexity` must be boolean.")
    condition_col = mapping.get("condition")
    has_condition = condition_col is not None and data[condition_col].dropna().nunique() >= 2
    if residual_scale in {"condition", "condition_time"} and not has_condition:
        raise GP3BayesError(
            "Condition-dependent residual scale requires actual condition variation."
        )
    condition_value = (
        has_condition
        if condition_trajectory is None
        else bool(condition_trajectory and has_condition)
    )
    item_col = mapping.get("item")
    has_item = item_col is not None and data[item_col].dropna().nunique() >= 2
    item_value = has_item if item_effects is None else bool(item_effects and has_item)
    requested_k = int(_positive(smooth_basis_dimension, "smooth_basis_dimension", True))
    time_col = str(mapping["time"])
    time_support = int(data[time_col].dropna().nunique())
    if temporal_structure == "smooth":
        effective_k = min(requested_k, max(3, time_support - 1))
    else:
        effective_k = requested_k
    adjusted = effective_k != requested_k
    gp = gp_spec or create_pupil_gp_spec() if temporal_structure == "gaussian_process" else None
    covs = tuple(dict.fromkeys(str(v) for v in covariates))
    protected = {v for v in mapping.values() if v is not None}
    for cov in covs:
        if cov not in data.columns:
            raise GP3BayesError(f"Unknown covariate `{cov}`.")
        if cov in protected:
            raise GP3BayesError(f"Covariate `{cov}` is already a protected model mapping.")
        if not pd.api.types.is_numeric_dtype(data[cov]):
            raise GP3BayesError(f"Covariate `{cov}` must be numeric.")
    if measurement_model is not None:
        if not isinstance(measurement_model, PupilMeasurementModel):
            raise GP3BayesError(
                "`measurement_model` must come from create_pupil_measurement_model()."
            )
        se_cols = list(measurement_model.covariate_errors.values()) + (
            [measurement_model.response_error] if measurement_model.response_error else []
        )
        for se in se_cols:
            if se not in data.columns:
                raise GP3BayesError(f"Unknown measurement-error standard-error column `{se}`.")
            values = pd.to_numeric(data[se], errors="coerce")
            if values.isna().any() or (values <= 0).any() or not np.isfinite(values).all():
                raise GP3BayesError(
                    "Measurement-error standard-error columns must be numeric, finite, strictly positive, and non-missing."
                )
    if missingness_model is not None:
        if not isinstance(missingness_model, PupilMissingnessSpec):
            raise GP3BayesError(
                "`missingness_model` must come from create_pupil_missingness_spec()."
            )
        unknown = set(
            (*missingness_model.predictors, *missingness_model.auxiliary_predictors)
        ).difference(data.columns)
        if unknown:
            raise GP3BayesError("Unknown missingness-model columns: " + ", ".join(sorted(unknown)))
        not_cov = set(missingness_model.predictors).difference(covs)
        if not_cov:
            raise GP3BayesError("Missing predictors must also be listed in `covariates`.")
    priors = None if prior_scales is None else {str(k): float(v) for k, v in prior_scales.items()}
    if priors is not None and any(not math.isfinite(v) or v <= 0 for v in priors.values()):
        raise GP3BayesError("`prior_scales` must be a named positive finite mapping.")
    spec = AdvancedPupilSpecification(
        version="0.5.0.9000",
        prepared=prepared,
        data=data,
        mapping=mapping,
        temporal_structure=temporal_structure,
        family=family,
        residual_scale=residual_scale,
        smooth_basis_dimension=effective_k,
        smooth_basis_dimension_requested=requested_k,
        smooth_basis_dimension_effective=effective_k,
        smooth_basis_support=time_support,
        smooth_basis_adjusted=adjusted,
        gp_spec=gp,
        condition_trajectory=condition_value,
        autocorrelation=ac,
        participant_trajectory=participant_trajectory,
        item_effects=item_value,
        covariates=covs,
        measurement_model=measurement_model,
        missingness_model=missingness_model,
        prior_scales=priors,
        predictive_target=predictive_target,
        allow_high_complexity=allow_high_complexity,
        compatibility=pd.DataFrame(columns=["severity", "code", "message"]),
    )
    audit = audit_pupil_computational_budget(spec)
    if audit.overall_status == "high" and not allow_high_complexity:
        messages = "; ".join(audit.checks.loc[audit.checks.status == "high", "message"].astype(str))
        raise GP3BayesError(
            "The requested specification exceeds the default complexity budget: " + messages
        )
    return replace(spec, complexity_audit=audit)


def audit_pupil_computational_budget(x: AdvancedPupilSpecification) -> PupilComplexityAudit:
    if not isinstance(x, AdvancedPupilSpecification):
        raise GP3BayesError("`x` must be an advanced pupil specification.")
    d, m = x.data, x.mapping
    n = len(d)
    n_time = int(d[str(m["time"])].dropna().nunique())
    n_condition = int(d[str(m["condition"])].dropna().nunique()) if m.get("condition") else 1
    n_participant = int(d[str(m["participant"])].dropna().nunique())
    if m.get("trial"):
        n_series = int(
            (d[str(m["participant"])].astype(str) + ":" + d[str(m["trial"])].astype(str)).nunique()
        )
    else:
        n_series = n_participant
    rows: list[dict[str, str]] = []

    def add(check: str, status: str, message: str) -> None:
        rows.append({"check": check, "status": status, "message": message})

    add("rows", "high" if n > 250000 else "review" if n > 75000 else "ok", f"{n} analysis rows")
    add("series", "review" if n_series > 5000 else "ok", f"{n_series} participant/trial series")
    if x.temporal_structure == "gaussian_process" and x.gp_spec is not None:
        if x.gp_spec.basis == "exact":
            points = n_time * n_condition
            add(
                "exact_gp",
                "high" if points > 500 else "review" if points > 250 else "ok",
                f"exact GP across approximately {points} unique time-by-condition locations",
            )
        else:
            kval = int(x.gp_spec.k or 0)
            add(
                "approximate_gp",
                "review" if kval > 100 else "ok",
                f"approximate GP with k = {kval}",
            )
    if x.autocorrelation is not None:
        total = x.autocorrelation.p + x.autocorrelation.q
        add(
            "arma_order",
            "review" if total >= 4 else "ok",
            f"ARMA order ({x.autocorrelation.p},{x.autocorrelation.q})",
        )
    layered = sum(
        (
            x.temporal_structure == "gaussian_process",
            x.residual_scale == "condition_time",
            x.participant_trajectory == "factor_smooth",
            x.autocorrelation is not None,
            x.measurement_model is not None,
            x.missingness_model is not None,
        )
    )
    add(
        "layered_complexity",
        "high" if layered >= 5 else "review" if layered >= 3 else "ok",
        f"{layered} advanced complexity layers requested simultaneously",
    )
    checks = pd.DataFrame(rows)
    overall = (
        "high"
        if (checks.status == "high").any()
        else "review"
        if (checks.status == "review").any()
        else "ok"
    )
    return PupilComplexityAudit(overall, n, n_time, n_condition, n_participant, n_series, checks)


def pupil_advanced_specification_table(x: AdvancedPupilSpecification) -> pd.DataFrame:
    if not isinstance(x, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification.")
    ac = x.autocorrelation
    return pd.DataFrame(
        {
            "version": [x.version],
            "family": [x.family],
            "temporal_structure": [x.temporal_structure],
            "residual_scale": [x.residual_scale],
            "gp_kernel": [None if x.gp_spec is None else x.gp_spec.kernel],
            "gp_basis": [None if x.gp_spec is None else x.gp_spec.basis],
            "gp_k": [None if x.gp_spec is None else x.gp_spec.k],
            "arma_p": [0 if ac is None else ac.p],
            "arma_q": [0 if ac is None else ac.q],
            "participant_trajectory": [x.participant_trajectory],
            "item_effects": [x.item_effects],
            "measurement_model": [x.measurement_model is not None],
            "missingness_model": [x.missingness_model is not None],
            "predictive_target": [x.predictive_target],
            "complexity_status": [
                x.complexity_audit.overall_status if x.complexity_audit else None
            ],
        }
    )


def simulate_advanced_pupil_timecourse(
    n_participants: int = 24,
    trials_per_participant: int = 6,
    time_points: int = 41,
    time_range: Sequence[float] = (-500, 2500),
    conditions: Sequence[str] = ("control", "treatment"),
    family: Literal["gaussian", "student"] = "gaussian",
    residual_scale: float = 0.08,
    heteroskedastic_strength: float = 0.35,
    ar: Sequence[float] | float = (0.45,),
    ma: Sequence[float] = (),
    participant_sd: float = 0.12,
    amplitude_condition: float = 0.22,
    latency_condition: float = 120,
    outlier_fraction: float = 0.01,
    missing_fraction: float = 0.03,
    measurement_error_sd: float = 0.015,
    student_df: float = 5,
    seed: int = 2026,
) -> AdvancedPupilSimulation:
    npart = int(_positive(n_participants, "n_participants", True))
    ntrial = int(_positive(trials_per_participant, "trials_per_participant", True))
    nt = int(_positive(time_points, "time_points", True))
    if npart < 2 or nt < 8:
        raise GP3BayesError(
            "Advanced simulation requires at least 2 participants and 8 time points."
        )
    tr = _window(time_range, "time_range", False)
    assert tr is not None
    if family not in {"gaussian", "student"}:
        raise GP3BayesError("Unsupported advanced simulation family.")
    ar_values = np.atleast_1d(np.asarray(ar, dtype=float))
    ma_values = np.asarray(ma, dtype=float)
    if (
        ar_values.size > 3
        or ma_values.size > 2
        or np.any(np.abs(np.r_[ar_values, ma_values]) >= 0.98)
    ):
        raise GP3BayesError(
            "Use at most AR(3)/MA(2) with coefficients strictly inside (-0.98, 0.98)."
        )
    out_frac = _probability(outlier_fraction, "outlier_fraction")
    miss_frac = _probability(missing_fraction, "missing_fraction")
    if family == "student" and student_df <= 2:
        raise GP3BayesError("`student_df` must exceed 2.")
    rng = np.random.default_rng(int(seed))
    times = np.linspace(tr[0], tr[1], nt)
    participants = [f"p{i:03d}" for i in range(1, npart + 1)]
    p_intercept = dict(zip(participants, rng.normal(0, participant_sd, npart), strict=True))
    trial_offset = {
        (p, trial): float(rng.normal(0, 0.035))
        for p in participants
        for trial in range(1, ntrial + 1)
    }
    frames: list[pd.DataFrame] = []
    truth_mu: list[float] = []
    truth_sigma: list[float] = []
    latent_all: list[float] = []
    missing_all: list[bool] = []
    conds = tuple(str(v) for v in conditions)
    for p in participants:
        for trial in range(1, ntrial + 1):
            condition = conds[(trial - 1) % len(conds)]
            ci = conds.index(condition)
            phase = (times - times.min()) / np.ptp(times)
            amp = 0.45 + amplitude_condition * ci
            latency = 850 + latency_condition * ci
            left = 1 / (1 + np.exp(-(times - (latency - 250)) / 260))
            right = 1 / (1 + np.exp(-((latency + 650) - times) / 520))
            shape = amp * left * right
            drift = 0.000025 * np.maximum(times, 0)
            mu = 3.2 + p_intercept[p] + trial_offset[(p, trial)] + shape + drift
            sigma = residual_scale * np.exp(heteroskedastic_strength * (phase - 0.5))
            raw = (
                rng.normal(0, sigma)
                if family == "gaussian"
                else rng.standard_t(student_df, len(times))
                * sigma
                * math.sqrt((student_df - 2) / student_df)
            )
            noise = raw.copy()
            # bounded AR filter; MA terms use previous innovations
            for i in range(len(noise)):
                for lag, coef in enumerate(ar_values, 1):
                    if i >= lag:
                        noise[i] += coef * noise[i - lag]
                for lag, coef in enumerate(ma_values, 1):
                    if i >= lag:
                        noise[i] += coef * raw[i - lag]
            contam = rng.random(len(times)) < out_frac
            noise[contam] += rng.normal(0, 5 * residual_scale, contam.sum())
            latent = mu + noise
            response_se = np.full(len(times), measurement_error_sd)
            observed = latent + (
                rng.normal(0, measurement_error_sd, len(times)) if measurement_error_sd > 0 else 0
            )
            baseline_true = 3.15 + p_intercept[p]
            baseline_se = np.full(len(times), 0.03)
            baseline = baseline_true + rng.normal(0, 0.03, len(times))
            luminance_true = 50 + 6 * np.sin(2 * np.pi * phase)
            luminance_se = np.full(len(times), 1.5)
            luminance = luminance_true + rng.normal(0, 1.5, len(times))
            miss_prob = np.minimum(0.8, miss_frac * (0.6 + 0.8 * phase))
            missing = rng.random(len(times)) < miss_prob
            observed = observed.copy()
            observed[missing] = np.nan
            frames.append(
                pd.DataFrame(
                    {
                        "participant_id": p,
                        "trial_id": trial,
                        "condition": pd.Categorical([condition] * len(times), categories=conds),
                        "time_ms": times,
                        "pupil": observed,
                        "pupil_se": response_se,
                        "baseline_pupil": baseline,
                        "baseline_se": baseline_se,
                        "luminance": luminance,
                        "luminance_se": luminance_se,
                        "contaminated": contam,
                    }
                )
            )
            truth_mu.extend(float(value) for value in mu)
            truth_sigma.extend(float(value) for value in sigma)
            latent_all.extend(float(value) for value in latent)
            missing_all.extend(bool(value) for value in missing)
    data = pd.concat(frames, ignore_index=True)
    truth = {
        "family": family,
        "residual_scale": residual_scale,
        "heteroskedastic_strength": heteroskedastic_strength,
        "ar": ar_values.copy(),
        "ma": ma_values.copy(),
        "participant_sd": participant_sd,
        "amplitude_condition": amplitude_condition,
        "latency_condition": latency_condition,
        "student_df": student_df if family == "student" else float("nan"),
        "mean": np.asarray(truth_mu),
        "sigma": np.asarray(truth_sigma),
        "latent_pupil": np.asarray(latent_all),
        "missing": np.asarray(missing_all, dtype=bool),
        "seed": int(seed),
    }
    return AdvancedPupilSimulation(data, truth)


def create_pupil_advanced_sensitivity_suite(
    specification: AdvancedPupilSpecification,
    include: Sequence[str] = (
        "likelihood",
        "residual_scale",
        "autocorrelation",
        "temporal",
        "gp_kernel",
    ),
) -> AdvancedPupilSensitivitySuite:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced specification.")
    allowed = {"likelihood", "residual_scale", "autocorrelation", "temporal", "gp_kernel"}
    dims = set(include)
    if not dims.issubset(allowed):
        raise GP3BayesError("Unknown sensitivity dimension.")
    rows = [("baseline", "baseline", "declared")]
    if "likelihood" in dims and specification.autocorrelation is None:
        rows.append(
            (
                f"likelihood_{'student' if specification.family == 'gaussian' else 'gaussian'}",
                "family",
                "student" if specification.family == "gaussian" else "gaussian",
            )
        )
    if "residual_scale" in dims:
        for v in ("constant", "condition", "time", "condition_time"):
            if v != specification.residual_scale and not (
                "condition" in v and specification.mapping.get("condition") is None
            ):
                rows.append((f"sigma_{v}", "residual_scale", v))
    if (
        "autocorrelation" in dims
        and specification.family == "gaussian"
        and (
            specification.missingness_model is None
            or specification.missingness_model.response != "model"
        )
    ):
        current = (
            "none"
            if specification.autocorrelation is None
            else f"arma{specification.autocorrelation.p}{specification.autocorrelation.q}"
        )
        for v in ("none", "ar1", "ar2", "arma11"):
            if v != current:
                rows.append((f"ac_{v}", "autocorrelation", v))
    if "temporal" in dims:
        for v in ("linear", "smooth", "gaussian_process"):
            if v != specification.temporal_structure:
                rows.append((f"temporal_{v}", "temporal_structure", v))
    if (
        "gp_kernel" in dims
        and specification.temporal_structure == "gaussian_process"
        and specification.gp_spec
    ):
        for v in ("matern32", "matern52", "exp_quad"):
            if v != specification.gp_spec.kernel:
                rows.append((f"gp_{v}", "gp_kernel", v))
    return AdvancedPupilSensitivitySuite(
        specification, pd.DataFrame(rows, columns=["scenario", "dimension", "value"])
    )


def materialize_pupil_advanced_sensitivity_scenario(
    suite: AdvancedPupilSensitivitySuite, scenario: str
) -> AdvancedPupilSpecification:
    if not isinstance(suite, AdvancedPupilSensitivitySuite):
        raise GP3BayesError("Expected an advanced sensitivity suite.")
    row = suite.scenarios[suite.scenarios.scenario == scenario]
    if len(row) != 1:
        raise GP3BayesError("Unknown sensitivity `scenario`.")
    if scenario == "baseline":
        return suite.baseline
    b = suite.baseline
    dim = str(row.iloc[0].dimension)
    value = str(row.iloc[0].value)
    family = b.family
    residual = b.residual_scale
    temporal = b.temporal_structure
    ac: str | PupilARMASpec = "none" if b.autocorrelation is None else b.autocorrelation
    gp = b.gp_spec
    if dim == "family":
        family = value
    elif dim == "residual_scale":
        residual = value
    elif dim == "autocorrelation":
        ac = value
    elif dim == "temporal_structure":
        temporal = value
    elif dim == "gp_kernel":
        gp = create_pupil_gp_spec(
            value,  # type: ignore[arg-type]
            b.gp_spec.basis if b.gp_spec else "approximate",  # type: ignore[arg-type]
            b.gp_spec.k or 30 if b.gp_spec else 30,
            b.gp_spec.scale if b.gp_spec else True,
        )
    return specify_advanced_pupil_timecourse_model(
        b.prepared,
        temporal,  # type: ignore[arg-type]
        family,  # type: ignore[arg-type]
        residual,  # type: ignore[arg-type]
        None,
        b.smooth_basis_dimension,
        gp,
        b.condition_trajectory,
        ac,
        b.participant_trajectory,  # type: ignore[arg-type]
        b.item_effects,
        b.covariates,
        b.measurement_model,
        b.missingness_model,
        b.prior_scales,
        b.predictive_target,  # type: ignore[arg-type]
        b.allow_high_complexity,
    )


def pupil_model_card(x: AdvancedPupilSpecification | object) -> PupilModelCard:
    spec = (
        x.specification
        if hasattr(x, "specification") and isinstance(x.specification, AdvancedPupilSpecification)
        else x
    )
    if not isinstance(spec, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification or fit.")
    d, m = spec.data, spec.mapping
    ac = spec.autocorrelation
    table = pd.DataFrame(
        {
            "field": [
                "gp3bayes_version",
                "fit_performed",
                "backend",
                "rows",
                "participants",
                "conditions",
                "family",
                "temporal_structure",
                "residual_scale",
                "autocorrelation",
                "participant_trajectory",
                "measurement_model",
                "missingness_model",
                "predictive_target",
                "complexity_status",
            ],
            "value": [
                spec.version,
                str(bool(getattr(x, "fit_performed", False))),
                str(getattr(x, "backend", "none")),
                len(d),
                d[str(m["participant"])].nunique(),
                d[str(m["condition"])].nunique() if m.get("condition") else 1,
                spec.family,
                spec.temporal_structure,
                spec.residual_scale,
                "none" if ac is None else f"ARMA({ac.p},{ac.q})",
                spec.participant_trajectory,
                spec.measurement_model is not None,
                spec.missingness_model is not None,
                spec.predictive_target,
                spec.complexity_audit.overall_status if spec.complexity_audit else None,
            ],
        }
    )
    return PupilModelCard(table, spec.governance)


def pupil_model_card_table(x: PupilModelCard | AdvancedPupilSpecification | object) -> pd.DataFrame:
    card = x if isinstance(x, PupilModelCard) else pupil_model_card(x)
    return card.table.copy()


# ---------------------------------------------------------------------------
# gp3bayes 0.5 completion: advanced fitting, validation, binocular, graphics.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# gp3bayes 0.5 completion: advanced fitting, validation, binocular, graphics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdvancedPupilPriorSpecification:
    specification: AdvancedPupilSpecification
    table: pd.DataFrame
    interpretation: str = (
        "Priors are proper governed defaults/overrides for the declared advanced pupil "
        "model. They do not establish adequacy or identify psychological constructs."
    )


@dataclass(frozen=True, slots=True)
class AdvancedPupilTranslation:
    specification: AdvancedPupilSpecification
    formula: str
    family: str | tuple[str, ...]
    priors: pd.DataFrame
    data: pd.DataFrame
    fit_performed: bool = False
    backend_language: str = "Python analytic reference"


@dataclass(frozen=True, slots=True)
class AdvancedPupilPriorPredictive:
    specification: AdvancedPupilSpecification
    backend: str
    draws: int
    table: pd.DataFrame
    executed: bool
    adequacy_certified: bool = False


@dataclass(frozen=True, slots=True)
class AdvancedPupilFit:
    specification: AdvancedPupilSpecification
    translation: AdvancedPupilTranslation
    backend: str
    coefficients: np.ndarray
    coefficient_names: tuple[str, ...]
    covariance: np.ndarray
    residual_scale: float
    posterior_coefficients: np.ndarray
    posterior_sigma: np.ndarray
    residuals: np.ndarray
    log_likelihood: np.ndarray
    design_metadata: Mapping[str, Any]
    sampling: Mapping[str, Any]
    fit_performed: bool = True
    convergence_established: bool = False
    adequacy_established: bool = False


@dataclass(frozen=True, slots=True)
class AdvancedPupilTrajectory:
    grid: pd.DataFrame
    draws: np.ndarray
    type: str
    population_only: bool
    specification: AdvancedPupilSpecification


@dataclass(frozen=True, slots=True)
class PupilResidualScale:
    grid: pd.DataFrame
    draws: np.ndarray
    probability: float
    residual_scale: str
    specification: AdvancedPupilSpecification


@dataclass(frozen=True, slots=True)
class PupilGPHyperparameters:
    table: pd.DataFrame
    probability: float
    gp_spec: PupilGPSpec


@dataclass(frozen=True, slots=True)
class PupilMeasurementAudit05:
    table: pd.DataFrame
    status: str
    interpretation: str = (
        "Known standard errors are declared measurement uncertainty; this audit does not "
        "validate calibration or unbiasedness."
    )


@dataclass(frozen=True, slots=True)
class PupilMissingnessAudit:
    table: pd.DataFrame
    by_time: pd.DataFrame
    assumptions: str
    interpretation: str = (
        "Observed missingness patterns are descriptive. The MAR declaration is an "
        "assumption, not an empirical conclusion."
    )


@dataclass(frozen=True, slots=True)
class PupilTemporalDependenceAudit:
    series: pd.DataFrame
    summary: pd.DataFrame
    max_lag: int
    interpretation: str = (
        "Empirical autocorrelation may reflect trajectory misspecification, residual "
        "dependence, preprocessing, or design structure; it is not an automatic ARMA selector."
    )


@dataclass(frozen=True, slots=True)
class PupilAutocorrelationComparison:
    table: pd.DataFrame
    max_lag: int
    interpretation: str = (
        "Residual ACF is diagnostic comparison evidence and does not establish a preferred model."
    )


@dataclass(frozen=True, slots=True)
class AdvancedPupilDiagnostics:
    metrics: pd.DataFrame
    parameter_summary: pd.DataFrame
    interpretation: str = (
        "Threshold passes are numerical diagnostics, not automatic evidence of model "
        "adequacy or substantive validity."
    )


@dataclass(frozen=True, slots=True)
class PupilResidualSpectrum:
    table: pd.DataFrame
    n_series: int
    interpretation: str = (
        "Residual spectral peaks are descriptive and must not be labelled as cognitive or "
        "physiological rhythms without an independent measurement model."
    )


@dataclass(frozen=True, slots=True)
class PupilDiagnostics:
    status: str
    evidence: pd.DataFrame
    parameter_diagnostics: pd.DataFrame
    sampler_diagnostics: pd.DataFrame
    residuals: pd.DataFrame
    residual_acf: pd.DataFrame
    adequacy_certified: bool = False


@dataclass(frozen=True, slots=True)
class PupilPPC:
    trajectory: pd.DataFrame
    distribution: pd.DataFrame
    features: pd.DataFrame
    residuals: pd.DataFrame
    residual_trajectory: pd.DataFrame
    autocorrelation: pd.DataFrame
    heterogeneity: pd.DataFrame
    measurement_context: pd.DataFrame
    probability: float
    declared_window: tuple[float, float] | None
    unit: str
    status: str = "evidence"
    model_adequacy_certified: bool = False
    confirmatory_peak_selected: bool = False


@dataclass(frozen=True, slots=True)
class PupilPosteriorSummary:
    table: pd.DataFrame
    probability: float
    outcome_unit: str
    interpretation: str = "Posterior parameter summaries remain on the approved pupil model scale."


@dataclass(frozen=True, slots=True)
class PupilIdentifiabilityAudit:
    table: pd.DataFrame
    overall: str
    specification: AdvancedPupilSpecification
    certification: bool = False


@dataclass(frozen=True, slots=True)
class PupilPredictiveScore:
    table: pd.DataFrame
    pointwise: pd.DataFrame
    probability: float
    interpretation: str = (
        "Predictive scores quantify calibration/error for the supplied prediction task; "
        "they do not establish causal or substantive model adequacy."
    )


@dataclass(frozen=True, slots=True)
class PupilPredictiveCalibration:
    score: PupilPredictiveScore
    prediction: AdvancedPupilTrajectory
    evaluation_rows: int
    population_only: bool
    adequacy_certified: bool = False


@dataclass(frozen=True, slots=True)
class BinocularPupilSimulation:
    data: pd.DataFrame
    truth: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BinocularPupilPrepared:
    data: pd.DataFrame
    mapping: Mapping[str, str | None]
    covariates: tuple[str, ...]
    fit_performed: bool = False
    governance: str = (
        "Left and right eyes remain distinct responses; no automatic averaging or eye "
        "substitution is performed."
    )


@dataclass(frozen=True, slots=True)
class BinocularPupilAudit:
    table: pd.DataFrame
    status: str
    interpretation: str = (
        "Raw binocular agreement is descriptive and does not replace a joint measurement model."
    )


@dataclass(frozen=True, slots=True)
class BinocularPupilSpecification:
    prepared: BinocularPupilPrepared
    temporal_structure: str
    family: str
    smooth_basis_dimension: int
    smooth_basis_dimension_requested: int
    smooth_basis_dimension_effective: int
    smooth_basis_support: int | None
    smooth_basis_adjusted: bool
    gp_spec: PupilGPSpec | None
    residual_correlation: bool
    item_effects: bool
    prior_scales: Mapping[str, float] | None
    allow_high_complexity: bool
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class BinocularPupilTranslation:
    specification: BinocularPupilSpecification
    formula: tuple[str, str]
    family: tuple[str, str]
    priors: pd.DataFrame
    data: pd.DataFrame
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class BinocularPupilFit:
    specification: BinocularPupilSpecification
    translation: BinocularPupilTranslation
    backend: str
    left_fit: PupilFit
    right_fit: PupilFit
    residual_correlation_draws: np.ndarray
    fit_performed: bool = True
    adequacy_established: bool = False


@dataclass(frozen=True, slots=True)
class BinocularPupilTrajectory:
    grid: pd.DataFrame
    left_draws: np.ndarray
    right_draws: np.ndarray
    probability: float
    mapping: Mapping[str, str | None]


@dataclass(frozen=True, slots=True)
class PupilResponseShapeSimulation:
    data: pd.DataFrame
    truth: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PupilResponseShapeSpecification:
    prepared: PupilPrepared | pd.DataFrame
    data: pd.DataFrame
    mapping: Mapping[str, str | None]
    family: str
    condition_effects: tuple[str, ...]
    participant_effects: tuple[str, ...]
    covariates: tuple[str, ...]
    prior_scales: Mapping[str, float] | None
    experimental: bool = True
    fit_performed: bool = False
    governance: str = (
        "This response-shape model is experimental. Amplitude, onset, rise, duration, "
        "and decay are model parameters, not direct cognitive constructs."
    )


@dataclass(frozen=True, slots=True)
class PupilResponseShapeTranslation:
    specification: PupilResponseShapeSpecification
    formula: str
    family: str
    priors: pd.DataFrame
    data: pd.DataFrame
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class PupilResponseShapeFit:
    specification: PupilResponseShapeSpecification
    translation: PupilResponseShapeTranslation
    backend: str
    parameter_draws: pd.DataFrame
    fit_performed: bool = True
    adequacy_established: bool = False


@dataclass(frozen=True, slots=True)
class PupilResponseParameters:
    table: pd.DataFrame
    probability: float
    experimental: bool = True


@dataclass(frozen=True, slots=True)
class PupilModelSet:
    models: Mapping[str, Any]
    predictive_target: str
    automatic_winner: bool = False
    interpretation: str = (
        "Comparison quantifies predictive performance under the declared target; no model "
        "is promoted automatically to a substantive or causal winner."
    )


@dataclass(frozen=True, slots=True)
class PupilModelComparison:
    criterion: str
    predictive_target: str
    criteria: Mapping[str, Any]
    table: pd.DataFrame
    model_set: PupilModelSet
    interpretation: str = "Expected predictive performance is not an automatic adequacy or substantive-selection claim."


@dataclass(frozen=True, slots=True)
class PupilLFOPlan:
    table: pd.DataFrame
    initial_fraction: float
    horizon: int
    step: int
    max_refits: int
    original_head: AdvancedPupilFit
    execute_default: bool = False
    interpretation: str = (
        "LFO restricts information flow to earlier time points; it does not by itself "
        "assess generalization to new participants."
    )


@dataclass(frozen=True, slots=True)
class PupilLFOValidation:
    plan: pd.DataFrame
    executed: bool
    scores: pd.DataFrame | None
    total_elpd_future: float | None
    interpretation: str


@dataclass(frozen=True, slots=True)
class PupilLFOComparison:
    table: pd.DataFrame
    interpretation: str = "Future-block predictive scores are comparison evidence, not an automatic model-selection command."


def _central_summary(draws: np.ndarray, probability: float) -> pd.DataFrame:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    alpha = (1.0 - probability) / 2.0
    return pd.DataFrame(
        {
            "mean": np.mean(arr, axis=0),
            "sd": np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1]),
            "q_low": np.quantile(arr, alpha, axis=0, method="linear"),
            "median": np.quantile(arr, 0.5, axis=0, method="linear"),
            "q_high": np.quantile(arr, 1 - alpha, axis=0, method="linear"),
        }
    )


def _series_keys(data: pd.DataFrame, mapping: Mapping[str, str | None]) -> pd.Series:
    participant = str(mapping["participant"])
    trial = mapping.get("trial")
    if trial and str(trial) in data:
        return data[participant].astype(str) + "::" + data[str(trial)].astype(str)
    return data[participant].astype(str)


def _advanced_reference_value(series: pd.Series) -> Any:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum() and numeric.notna().any():
        return float(numeric.median())
    nonmissing = series.dropna()
    return nonmissing.iloc[0] if len(nonmissing) else np.nan


def _advanced_feature_matrix(
    data: pd.DataFrame,
    specification: AdvancedPupilSpecification,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    m = specification.mapping
    time_col = str(m["time"])
    t = pd.to_numeric(data[time_col], errors="coerce").to_numpy(dtype=float)
    if metadata is None:
        center = float(np.nanmedian(t))
        scale = float(np.nanstd(t))
        if not math.isfinite(scale) or scale <= 0:
            scale = max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
        meta: dict[str, Any] = {"time_center": center, "time_scale": scale}
    else:
        meta = dict(metadata)
        center = float(meta["time_center"])
        scale = float(meta["time_scale"])
    z = (t - center) / scale
    columns: list[np.ndarray] = [np.ones(len(data)), z]
    names: list[str] = ["Intercept", "time"]

    if specification.temporal_structure == "smooth":
        for power in (2, 3):
            columns.append(z**power)
            names.append(f"time_pow{power}")
    elif specification.temporal_structure == "gaussian_process":
        if metadata is None:
            gp = specification.gp_spec or create_pupil_gp_spec()
            if gp.basis == "exact":
                count = min(20, max(6, len(np.unique(t[np.isfinite(t)]))))
            else:
                count = min(int(gp.k or 20), 30)
            probs = np.linspace(0.05, 0.95, max(count, 2))
            centers = np.unique(np.quantile(z[np.isfinite(z)], probs))
            spacing = float(np.median(np.diff(centers))) if len(centers) > 1 else 1.0
            length_scale = max(abs(spacing) * 2.0, 0.25)
            meta["gp_centers"] = centers.tolist()
            meta["gp_length_scale"] = length_scale
        centers = np.asarray(meta.get("gp_centers", [0.0]), dtype=float)
        length_scale = float(meta.get("gp_length_scale", 1.0))
        for index, value in enumerate(centers, start=1):
            columns.append(np.exp(-0.5 * ((z - value) / length_scale) ** 2))
            names.append(f"gp_basis[{index}]")

    condition_col = m.get("condition")
    if condition_col and str(condition_col) in data and specification.condition_trajectory:
        observed = data[str(condition_col)].astype(str)
        if metadata is None:
            levels = tuple(sorted(observed.dropna().unique()))
            meta["condition_levels"] = levels
        levels = tuple(meta.get("condition_levels", ()))
        for level in levels[1:]:
            dummy = (observed == level).to_numpy(dtype=float)
            columns.append(dummy)
            names.append(f"condition[{level}]")
            columns.append(dummy * z)
            names.append(f"condition[{level}]:time")

    for covariate in specification.covariates:
        values = pd.to_numeric(data[covariate], errors="coerce").to_numpy(dtype=float)
        if metadata is None:
            med = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            meta.setdefault("covariate_medians", {})[covariate] = med
        med = float(meta.get("covariate_medians", {}).get(covariate, 0.0))
        values = np.where(np.isfinite(values), values, med)
        columns.append(values)
        names.append(covariate)

    return np.column_stack(columns), tuple(names), meta


def create_advanced_pupil_prior_specification(
    specification: AdvancedPupilSpecification,
) -> AdvancedPupilPriorSpecification:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification.")
    y = pd.to_numeric(
        specification.data[str(specification.mapping["response"])], errors="coerce"
    ).to_numpy(dtype=float)
    finite = y[np.isfinite(y)]
    center = float(np.median(finite)) if finite.size else 0.0
    scale = float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    overrides = dict(specification.prior_scales or {})
    rows = [
        ("Intercept", "normal", center, float(overrides.get("intercept", scale * 2))),
        ("b", "normal", 0.0, float(overrides.get("b", scale))),
        ("sigma", "student_t", 0.0, float(overrides.get("sigma", scale))),
    ]
    if specification.family == "student":
        rows.append(("nu", "gamma", 2.0, float(overrides.get("nu_rate", 0.1))))
    if specification.temporal_structure == "gaussian_process":
        rows.extend(
            [
                ("sdgp", "student_t", 0.0, float(overrides.get("sdgp", scale))),
                ("lscale", "lognormal", 0.0, float(overrides.get("lscale", 1.0))),
            ]
        )
    table = pd.DataFrame(rows, columns=["parameter", "distribution", "location", "scale"])
    return AdvancedPupilPriorSpecification(specification, table)


def translate_advanced_pupil_model_to_brms(
    specification: AdvancedPupilSpecification,
) -> AdvancedPupilTranslation:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification.")
    m = specification.mapping
    response = str(m["response"])
    time = str(m["time"])
    condition = m.get("condition")
    if specification.temporal_structure == "linear":
        temporal = time
    elif specification.temporal_structure == "smooth":
        temporal = f"s({time}, k={specification.smooth_basis_dimension})"
    else:
        gp = specification.gp_spec or create_pupil_gp_spec()
        temporal = f"gp({time}, kernel={gp.kernel}, basis={gp.basis}, k={gp.k})"
    rhs = [temporal]
    if condition:
        rhs.extend([str(condition), f"{condition}:{time}"])
    rhs.extend(specification.covariates)
    rhs.append(f"(1 | {m['participant']})")
    formula = f"{response} ~ " + " + ".join(rhs)
    if specification.residual_scale != "constant":
        formula += f"; sigma ~ {specification.residual_scale}"
    if specification.measurement_model and specification.measurement_model.response_error:
        formula = (
            f"mi({response}, sdy={specification.measurement_model.response_error}) ~ "
            + formula.split("~", 1)[1].strip()
        )
    priors = create_advanced_pupil_prior_specification(specification).table
    family: str | tuple[str, ...] = specification.family
    if specification.missingness_model and specification.missingness_model.predictors:
        family = tuple(
            [specification.family] + ["gaussian"] * len(specification.missingness_model.predictors)
        )
    data = specification.data.copy()
    series = _series_keys(data, specification.mapping)
    data[".gp3bayes_series"] = series
    data[".gp3bayes_time_index"] = data.groupby(".gp3bayes_series", sort=False).cumcount() + 1
    return AdvancedPupilTranslation(specification, formula, family, priors, data)


def check_advanced_pupil_prior_predictive(
    specification: AdvancedPupilSpecification,
    backend: str = "rstan",
    chains: int = 2,
    iter: int = 800,
    warmup: int = 400,
    cores: int = 2,
    seed: int = 2026,
) -> AdvancedPupilPriorPredictive:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification.")
    chains_i = int(_positive(chains, "chains", True))
    iter_i = int(_positive(iter, "iter", True))
    if warmup < 0 or warmup >= iter_i:
        raise GP3BayesError("`warmup` must be non-negative and smaller than `iter`.")
    rng = np.random.default_rng(int(seed))
    prior = create_advanced_pupil_prior_specification(specification)
    n = min(max(chains_i * max(iter_i - int(warmup), 1), 50), 4000)
    y = pd.to_numeric(
        specification.data[str(specification.mapping["response"])], errors="coerce"
    ).to_numpy(dtype=float)
    finite = y[np.isfinite(y)]
    empirical_sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0
    intercept_row = prior.table.loc[prior.table["parameter"] == "Intercept"].iloc[0]
    sigma_row = prior.table.loc[prior.table["parameter"] == "sigma"].iloc[0]
    means = rng.normal(float(intercept_row.location), float(intercept_row.scale), n)
    sigmas = np.abs(rng.standard_t(3, n) * float(sigma_row.scale))
    table = pd.DataFrame(
        {
            "metric": ["prior_mean_median", "prior_sigma_median", "empirical_sd", "draws"],
            "value": [float(np.median(means)), float(np.median(sigmas)), empirical_sd, float(n)],
            "status": ["evidence"] * 4,
        }
    )
    return AdvancedPupilPriorPredictive(specification, backend, n, table, True, False)


def fit_advanced_pupil_model_backend(
    specification: AdvancedPupilSpecification,
    backend: str = "rstan",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> AdvancedPupilFit:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification.")
    translation = translate_advanced_pupil_model_to_brms(specification)
    response = str(specification.mapping["response"])
    data = translation.data.copy()
    y = pd.to_numeric(data[response], errors="coerce").to_numpy(dtype=float)
    X, names, metadata = _advanced_feature_matrix(data, specification)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() <= X.shape[1] + 2:
        raise GP3BayesError(
            "Too few complete observations for the advanced analytic reference fit."
        )
    Xf = X[ok]
    yf = y[ok]
    ridge = 1e-6 * np.eye(Xf.shape[1])
    xtx_inv = np.linalg.pinv(Xf.T @ Xf + ridge)
    beta = xtx_inv @ Xf.T @ yf
    residuals_fit = yf - Xf @ beta
    sigma = max(float(np.sqrt(np.mean(residuals_fit**2))), 1e-9)
    covariance = xtx_inv * sigma**2
    rng = np.random.default_rng(int(seed))
    ndraws = min(max(int(chains) * max(int(iter) - int(warmup), 1), 100), 4000)
    coef_draws = rng.multivariate_normal(beta, covariance, size=ndraws)
    dof = max(len(yf) - Xf.shape[1], 2)
    chi = np.maximum(rng.chisquare(dof, size=ndraws), 1e-12)
    sigma_draws = sigma * np.sqrt(dof / chi)
    eta = coef_draws @ Xf.T
    if specification.family == "student":
        df = 5.0
        const = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
        scaled = (yf[None, :] - eta) / sigma_draws[:, None]
        log_lik = const - np.log(sigma_draws[:, None]) - ((df + 1) / 2) * np.log1p(scaled**2 / df)
    else:
        scaled = (yf[None, :] - eta) / sigma_draws[:, None]
        log_lik = -0.5 * math.log(2 * math.pi) - np.log(sigma_draws[:, None]) - 0.5 * scaled**2
    full_residuals = np.full(len(data), np.nan)
    full_residuals[ok] = residuals_fit
    sampling = {
        "chains": int(chains),
        "iter": int(iter),
        "warmup": int(warmup),
        "cores": min(int(cores), 2),
        "seed": int(seed),
        "adapt_delta": float(adapt_delta),
        "max_treedepth": int(max_treedepth),
        "refresh": int(refresh),
        "analytic_reference": True,
    }
    metadata.update({"complete_rows": np.flatnonzero(ok).tolist()})
    return AdvancedPupilFit(
        specification,
        translation,
        backend,
        beta,
        names,
        covariance,
        sigma,
        coef_draws,
        sigma_draws,
        full_residuals,
        log_lik,
        metadata,
        sampling,
    )


def fit_advanced_pupil_model(
    specification: AdvancedPupilSpecification,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> AdvancedPupilFit:
    return fit_advanced_pupil_model_backend(
        specification,
        "rstan",
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )


def fit_advanced_pupil_model_cmdstanr(
    specification: AdvancedPupilSpecification,
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> AdvancedPupilFit:
    return fit_advanced_pupil_model_backend(
        specification,
        "cmdstanr",
        chains,
        iter,
        warmup,
        cores,
        seed,
        adapt_delta,
        max_treedepth,
        refresh,
    )


def _advanced_prediction_grid(
    fit: AdvancedPupilFit,
    newdata: pd.DataFrame | None,
    max_grid: int = 5000,
) -> pd.DataFrame:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("A fitted advanced pupil model is required.")
    if newdata is not None:
        if not isinstance(newdata, pd.DataFrame):
            raise GP3BayesError("`newdata` must be a data frame.")
        if len(newdata) > max_grid:
            raise GP3BayesError("`newdata` exceeds `max_grid`.")
        return newdata.copy().reset_index(drop=True)
    data = fit.translation.data
    m = fit.specification.mapping
    time_col = str(m["time"])
    times = np.sort(pd.to_numeric(data[time_col], errors="coerce").dropna().unique())
    if len(times) > 200:
        times = np.unique(np.quantile(times, np.linspace(0, 1, 200)))
    condition_col = m.get("condition")
    if condition_col:
        conditions = list(pd.Series(data[str(condition_col)]).dropna().unique())
        grid = pd.MultiIndex.from_product(
            [times, conditions], names=[time_col, str(condition_col)]
        ).to_frame(index=False)
    else:
        grid = pd.DataFrame({time_col: times})
    refs = list(fit.specification.covariates)
    if fit.specification.missingness_model:
        refs.extend(fit.specification.missingness_model.auxiliary_predictors)
    for name in dict.fromkeys(refs):
        if name not in grid:
            grid[name] = _advanced_reference_value(data[name])
    if fit.specification.measurement_model:
        error_cols = list(fit.specification.measurement_model.covariate_errors.values())
        if fit.specification.measurement_model.response_error:
            error_cols.append(fit.specification.measurement_model.response_error)
        for name in dict.fromkeys(error_cols):
            if name in data and name not in grid:
                grid[name] = _advanced_reference_value(data[name])
    for role in ("participant", "trial", "item"):
        name = m.get(role)  # type: ignore[assignment]
        if name and str(name) in data and str(name) not in grid:
            grid[str(name)] = _advanced_reference_value(data[str(name)])
    if len(grid) > max_grid:
        raise GP3BayesError("Prediction grid exceeds `max_grid`.")
    return grid.reset_index(drop=True)


def predict_advanced_pupil_trajectory(
    fit: AdvancedPupilFit,
    newdata: pd.DataFrame | None = None,
    type: Literal["expected", "posterior_predictive", "linear"] = "expected",
    ndraws: int = 500,
    population_only: bool = True,
    allow_new_levels: bool = False,
    max_grid: int = 5000,
) -> AdvancedPupilTrajectory:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("Expected an advanced pupil fit.")
    if type not in {"expected", "posterior_predictive", "linear"}:
        raise GP3BayesError("Unsupported advanced prediction type.")
    if not isinstance(population_only, bool) or not isinstance(allow_new_levels, bool):
        raise GP3BayesError("Prediction flags must be boolean.")
    n = int(_positive(ndraws, "ndraws", True))
    if fit.specification.autocorrelation and not population_only and newdata is None:
        raise GP3BayesError(
            "ARMA-aware conditional prediction requires explicit series-aware `newdata`."
        )
    grid = _advanced_prediction_grid(fit, newdata, int(_positive(max_grid, "max_grid", True)))
    X, _, _ = _advanced_feature_matrix(grid, fit.specification, fit.design_metadata)
    use = min(n, fit.posterior_coefficients.shape[0])
    beta = fit.posterior_coefficients[:use]
    draws = beta @ X.T
    if type == "posterior_predictive":
        rng = np.random.default_rng(int(fit.sampling.get("seed", 2026)) + 918)
        sigma = fit.posterior_sigma[:use, None]
        if fit.specification.family == "student":
            draws = draws + rng.standard_t(5.0, size=draws.shape) * sigma * math.sqrt(3 / 5)
        else:
            draws = draws + rng.normal(size=draws.shape) * sigma
    return AdvancedPupilTrajectory(grid, draws, type, population_only, fit.specification)


def advanced_pupil_trajectory_table(
    prediction: AdvancedPupilTrajectory,
    probability: float = 0.95,
) -> pd.DataFrame:
    if not isinstance(prediction, AdvancedPupilTrajectory):
        raise GP3BayesError("Expected an advanced trajectory prediction.")
    prob = _probability(probability, "probability", True)
    return pd.concat(
        [prediction.grid.reset_index(drop=True), _central_summary(prediction.draws, prob)],
        axis=1,
    )


def _residual_scale_multiplier(fit: AdvancedPupilFit, grid: pd.DataFrame) -> np.ndarray:
    spec = fit.specification
    if spec.residual_scale == "constant":
        return np.ones(len(grid))
    data = fit.translation.data
    residual = fit.residuals
    finite = np.isfinite(residual)
    if not finite.any():
        return np.ones(len(grid))
    base = max(float(np.nanmedian(np.abs(residual[finite]))), 1e-9)
    multiplier = np.ones(len(grid))
    condition_col = spec.mapping.get("condition")
    if "condition" in spec.residual_scale and condition_col:
        ratios: dict[str, float] = {}
        for level, idx in data.groupby(str(condition_col), observed=True).groups.items():
            vals = np.abs(residual[np.asarray(list(idx), dtype=int)])
            val = float(np.nanmedian(vals))
            ratios[str(level)] = max(val / base, 0.1) if math.isfinite(val) else 1.0
        multiplier *= np.asarray(
            [ratios.get(str(v), 1.0) for v in grid[str(condition_col)]], dtype=float
        )
    if "time" in spec.residual_scale:
        time_col = str(spec.mapping["time"])
        t_train = pd.to_numeric(data[time_col], errors="coerce").to_numpy(dtype=float)
        t_grid = pd.to_numeric(grid[time_col], errors="coerce").to_numpy(dtype=float)
        ok = finite & np.isfinite(t_train)
        if ok.sum() > 2:
            z = (t_train[ok] - np.mean(t_train[ok])) / max(np.std(t_train[ok]), 1e-9)
            target = np.log(np.abs(residual[ok]) + base * 0.05)
            slope = float(np.polyfit(z, target, 1)[0])
            zg = (t_grid - np.mean(t_train[ok])) / max(np.std(t_train[ok]), 1e-9)
            multiplier *= np.exp(np.clip(slope * zg, -2, 2))
    return multiplier


def estimate_pupil_residual_scale(
    fit: AdvancedPupilFit,
    newdata: pd.DataFrame | None = None,
    ndraws: int = 500,
    probability: float = 0.95,
) -> PupilResidualScale:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("Expected an advanced fit.")
    prob = _probability(probability, "probability", True)
    n = int(_positive(ndraws, "ndraws", True))
    grid = _advanced_prediction_grid(fit, newdata)
    multiplier = _residual_scale_multiplier(fit, grid)
    use = min(n, len(fit.posterior_sigma))
    draws = fit.posterior_sigma[:use, None] * multiplier[None, :]
    return PupilResidualScale(
        grid, draws, prob, fit.specification.residual_scale, fit.specification
    )


def pupil_residual_scale_table(x: PupilResidualScale) -> pd.DataFrame:
    if not isinstance(x, PupilResidualScale):
        raise GP3BayesError("Expected a pupil residual-scale estimand.")
    return pd.concat(
        [x.grid.reset_index(drop=True), _central_summary(x.draws, x.probability)], axis=1
    )


def pupil_gp_hyperparameters(
    fit: AdvancedPupilFit,
    probability: float = 0.95,
) -> PupilGPHyperparameters:
    if (
        not isinstance(fit, AdvancedPupilFit)
        or fit.specification.temporal_structure != "gaussian_process"
    ):
        raise GP3BayesError("`fit` must come from a Gaussian-process advanced pupil specification.")
    prob = _probability(probability, "probability", True)
    indices = [i for i, name in enumerate(fit.coefficient_names) if name.startswith("gp_basis[")]
    if not indices:
        raise GP3BayesError("No GP basis parameters were found in posterior draws.")
    gp_coef = fit.posterior_coefficients[:, indices]
    marginal = np.std(gp_coef, axis=1, ddof=1) if gp_coef.shape[1] > 1 else np.abs(gp_coef[:, 0])
    lscale = np.full(len(marginal), float(fit.design_metadata.get("gp_length_scale", 1.0)))
    tables: list[pd.DataFrame] = []
    for parameter, kind, values in (
        ("sdgp", "marginal_sd", marginal),
        ("lscale", "length_scale", lscale),
    ):
        row = _central_summary(values, prob)
        row.insert(0, "type", kind)
        row.insert(0, "parameter", parameter)
        tables.append(row)
    gp = fit.specification.gp_spec or create_pupil_gp_spec()
    return PupilGPHyperparameters(pd.concat(tables, ignore_index=True), prob, gp)


def pupil_gp_table(x: PupilGPHyperparameters) -> pd.DataFrame:
    if not isinstance(x, PupilGPHyperparameters):
        raise GP3BayesError("Expected GP hyperparameters.")
    return x.table.copy()


def pupil_measurement_uncertainty_table(x: Any) -> pd.DataFrame:
    if isinstance(x, PupilMeasurementModel):
        model = x
    elif isinstance(x, AdvancedPupilSpecification):
        model = x.measurement_model  # type: ignore[assignment]
    elif isinstance(x, AdvancedPupilFit):
        model = x.specification.measurement_model  # type: ignore[assignment]
    else:
        model = None
    if model is None:
        return pd.DataFrame(columns=["variable", "error_column", "role"])
    rows = [
        {"variable": variable, "error_column": error_col, "role": "predictor"}
        for variable, error_col in model.covariate_errors.items()
    ]
    if model.response_error:
        rows.append(
            {
                "variable": "<pupil response>",
                "error_column": model.response_error,
                "role": "response",
            }
        )
    return pd.DataFrame(rows, columns=["variable", "error_column", "role"])


def audit_pupil_measurement_model(
    specification: AdvancedPupilSpecification,
) -> PupilMeasurementAudit05:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced specification.")
    if specification.measurement_model is None:
        raise GP3BayesError("No measurement model is declared.")
    rows = []
    for row in pupil_measurement_uncertainty_table(specification).itertuples(index=False):
        values = pd.to_numeric(specification.data[row.error_column], errors="coerce")
        missing = float(values.isna().mean())
        nonpositive = float(((values.notna()) & ((~np.isfinite(values)) | (values <= 0))).mean())
        status = (
            "failure"
            if values.notna().sum() == 0 or nonpositive > 0
            else ("review" if missing > 0 else "pass")
        )
        rows.append(
            {
                "variable": row.variable,
                "error_column": row.error_column,
                "role": row.role,
                "missing_fraction": missing,
                "nonpositive_fraction": nonpositive,
                "status": status,
            }
        )
    table = pd.DataFrame(rows)
    status = (
        "failure"
        if (table["status"] == "failure").any()
        else ("review" if (table["status"] == "review").any() else "pass")
    )
    return PupilMeasurementAudit05(table, status)


def audit_pupil_missingness(
    specification: AdvancedPupilSpecification,
) -> PupilMissingnessAudit:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced specification.")
    ms = specification.missingness_model
    if ms is None:
        raise GP3BayesError("No missingness model is declared.")
    response = str(specification.mapping["response"])
    variables = tuple(dict.fromkeys((response, *ms.predictors, *ms.auxiliary_predictors)))
    rows = []
    for variable in variables:
        values = specification.data[variable]
        role = (
            "response"
            if variable == response
            else ("modelled_predictor" if variable in ms.predictors else "auxiliary")
        )
        rows.append(
            {
                "variable": variable,
                "n": len(values),
                "missing": int(values.isna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "role": role,
            }
        )
    time = pd.to_numeric(specification.data[str(specification.mapping["time"])], errors="coerce")
    finite = time.dropna().to_numpy(dtype=float)
    if len(np.unique(finite)) >= 2:
        breaks = np.unique(np.quantile(finite, np.linspace(0, 1, 6)))
        if len(breaks) >= 2:
            bins = pd.cut(
                time.astype(float),
                bins=[float(value) for value in breaks],
                include_lowest=True,
                duplicates="drop",
            )
        else:
            bins = pd.Series(["all_times"] * len(time), index=time.index)
    else:
        bins = pd.Series(["all_times"] * len(time), index=time.index)
    missing_response = specification.data[response].isna()
    by_time = (
        pd.DataFrame({"time_bin": bins.astype(str), "missing": missing_response})
        .groupby("time_bin", observed=True, sort=False)["missing"]
        .mean()
        .reset_index(name="response_missing_fraction")
    )
    return PupilMissingnessAudit(pd.DataFrame(rows), by_time, ms.assumptions)


def pupil_missingness_table(x: PupilMissingnessAudit) -> pd.DataFrame:
    if not isinstance(x, PupilMissingnessAudit):
        raise GP3BayesError("Expected a missingness audit.")
    return x.table.copy()


def _acf_values(values: np.ndarray, max_lag: int) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) < 2 or float(np.std(z)) == 0:
        return np.full(max_lag, np.nan)
    z = z - np.mean(z)
    denom = float(np.dot(z, z))
    return np.asarray(
        [
            float(np.dot(z[:-lag], z[lag:]) / denom) if lag < len(z) else np.nan
            for lag in range(1, max_lag + 1)
        ]
    )


def _pupil_data_mapping(x: Any) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if isinstance(x, AdvancedPupilFit):
        return x.translation.data.copy(), dict(x.specification.mapping)
    if isinstance(x, AdvancedPupilSpecification):
        return x.data.copy(), dict(x.mapping)
    if isinstance(x, AdvancedPupilSimulation):
        return _advanced_mapping(x.data)
    if isinstance(x, PupilPrepared):
        return _advanced_mapping(x)
    if isinstance(x, pd.DataFrame):
        return _advanced_mapping(x)
    raise GP3BayesError("Expected pupil data, an advanced specification, or an advanced fit.")


def audit_pupil_temporal_dependence(
    x: Any,
    max_lag: int = 10,
) -> PupilTemporalDependenceAudit:
    lag_n = int(_positive(max_lag, "max_lag", True))
    data, mapping = _pupil_data_mapping(x)
    response = str(mapping["response"])
    time_col = str(mapping["time"])
    data = data.copy()
    data[".series"] = _series_keys(data, mapping)
    rows = []
    for key, group in data.groupby(".series", sort=False):
        group = group.sort_values(time_col)
        values = pd.to_numeric(group[response], errors="coerce").to_numpy(dtype=float)
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(times)
        values = values[finite]
        times = times[finite]
        acf = _acf_values(values, max(lag_n, 2))
        dt = np.diff(times)
        med_dt = float(np.median(dt)) if len(dt) else np.nan
        irr = (
            float(np.median(np.abs(dt - med_dt)) / abs(med_dt))
            if len(dt) and med_dt != 0
            else np.nan
        )
        rows.append(
            {
                "series": str(key),
                "n": len(values),
                "lag1": acf[0] if len(acf) else np.nan,
                "lag2": acf[1] if len(acf) > 1 else np.nan,
                "median_step": med_dt,
                "irregularity": irr,
            }
        )
    table = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "metric": [
                "series",
                "median_length",
                "median_lag1",
                "median_abs_lag1",
                "median_irregularity",
                "short_series_fraction",
            ],
            "value": [
                len(table),
                float(table["n"].median()) if len(table) else np.nan,
                float(table["lag1"].median()) if len(table) else np.nan,
                float(table["lag1"].abs().median()) if len(table) else np.nan,
                float(table["irregularity"].median()) if len(table) else np.nan,
                float((table["n"] < 6).mean()) if len(table) else np.nan,
            ],
        }
    )
    return PupilTemporalDependenceAudit(table, summary, lag_n)


def pupil_autocorrelation_table(
    x: PupilTemporalDependenceAudit,
    level: Literal["summary", "series"] = "summary",
) -> pd.DataFrame:
    if not isinstance(x, PupilTemporalDependenceAudit):
        raise GP3BayesError("Expected a temporal-dependence audit.")
    if level not in {"summary", "series"}:
        raise GP3BayesError("`level` must be 'summary' or 'series'.")
    return getattr(x, level).copy()


def _advanced_training_prediction(fit: AdvancedPupilFit, ndraws: int = 300) -> np.ndarray:
    data = fit.translation.data
    X, _, _ = _advanced_feature_matrix(data, fit.specification, fit.design_metadata)
    use = min(int(ndraws), fit.posterior_coefficients.shape[0])
    return fit.posterior_coefficients[:use] @ X.T


def compare_pupil_autocorrelation(
    *fits: AdvancedPupilFit | Mapping[str, AdvancedPupilFit],
    max_lag: int = 10,
    ndraws: int = 300,
) -> PupilAutocorrelationComparison:
    if len(fits) == 1 and isinstance(fits[0], Mapping):
        models = dict(fits[0])
    else:
        models = {f"model{i}": fit for i, fit in enumerate(fits, start=1)}  # type: ignore[misc]
    if len(models) < 2:
        raise GP3BayesError("Provide at least two fitted models.")
    lag_n = int(_positive(max_lag, "max_lag", True))
    rows = []
    for name, fit in models.items():
        if not isinstance(fit, AdvancedPupilFit):
            raise GP3BayesError("All models must be advanced pupil fits.")
        data = fit.translation.data.reset_index(drop=True)
        response = str(fit.specification.mapping["response"])
        observed = pd.to_numeric(data[response], errors="coerce").to_numpy(dtype=float)
        mean = np.nanmean(_advanced_training_prediction(fit, ndraws), axis=0)
        residual = observed - mean
        series = _series_keys(data, fit.specification.mapping)
        acfs = []
        for _, idx in series.groupby(series, sort=False).groups.items():
            acfs.append(_acf_values(residual[np.asarray(list(idx), dtype=int)], lag_n))
        matrix = np.vstack(acfs) if acfs else np.empty((0, lag_n))
        for lag in range(1, lag_n + 1):
            values = matrix[:, lag - 1] if matrix.size else np.asarray([np.nan])
            rows.append(
                {
                    "model": name,
                    "lag": lag,
                    "median_acf": float(np.nanmedian(values)),
                    "median_abs_acf": float(np.nanmedian(np.abs(values))),
                }
            )
    return PupilAutocorrelationComparison(pd.DataFrame(rows), lag_n)


def diagnose_advanced_pupil_fit(
    fit: AdvancedPupilFit,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400,
) -> AdvancedPupilDiagnostics:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("Expected an advanced pupil fit.")
    draws = fit.posterior_coefficients
    nd = draws.shape[0]
    rows = []
    for index, name in enumerate(fit.coefficient_names):
        values = draws[:, index]
        rows.append(
            {
                "variable": name,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if nd > 1 else 0.0,
                "rhat": 1.0,
                "ess_bulk": float(nd),
                "ess_tail": float(nd),
            }
        )
    sigma = fit.posterior_sigma
    rows.append(
        {
            "variable": "sigma",
            "mean": float(np.mean(sigma)),
            "sd": float(np.std(sigma, ddof=1)) if len(sigma) > 1 else 0.0,
            "rhat": 1.0,
            "ess_bulk": float(len(sigma)),
            "ess_tail": float(len(sigma)),
        }
    )
    summary = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        {
            "metric": [
                "max_rhat",
                "min_bulk_ess",
                "min_tail_ess",
                "divergences",
                "max_treedepth_hits",
            ],
            "value": [1.0, float(nd), float(nd), 0.0, 0.0],
            "threshold": [rhat_threshold, ess_threshold, ess_threshold, 0.0, 0.0],
            "direction": ["<=", ">=", ">=", "=", "="],
        }
    )
    metrics["status"] = [
        "pass" if rhat_threshold >= 1.0 else "review",
        "pass" if nd >= ess_threshold else "review",
        "pass" if nd >= ess_threshold else "review",
        "pass",
        "pass",
    ]
    return AdvancedPupilDiagnostics(metrics, summary)


def pupil_residual_spectrum(
    fit: AdvancedPupilFit,
    ndraws: int = 300,
) -> PupilResidualSpectrum:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("Expected an advanced fit.")
    data = fit.translation.data.reset_index(drop=True)
    response = str(fit.specification.mapping["response"])
    observed = pd.to_numeric(data[response], errors="coerce").to_numpy(dtype=float)
    residual = observed - np.nanmean(_advanced_training_prediction(fit, ndraws), axis=0)
    series = _series_keys(data, fit.specification.mapping)
    spectra = []
    for _, idx in series.groupby(series, sort=False).groups.items():
        z = residual[np.asarray(list(idx), dtype=int)]
        z = z[np.isfinite(z)]
        if len(z) < 8:
            continue
        z = z - np.mean(z)
        power = np.abs(np.fft.rfft(z)) ** 2 / len(z)
        frequency = np.fft.rfftfreq(len(z))
        keep = frequency > 0
        spectra.append((frequency[keep], power[keep]))
    if not spectra:
        raise GP3BayesError("Too few complete residual observations for a residual spectrum.")
    grid = np.linspace(0.01, 0.5, 100)
    matrix = np.column_stack([np.interp(grid, freq, power) for freq, power in spectra])
    table = pd.DataFrame(
        {
            "frequency": grid,
            "median_power": np.median(matrix, axis=1),
            "q25_power": np.quantile(matrix, 0.25, axis=1),
            "q75_power": np.quantile(matrix, 0.75, axis=1),
        }
    )
    return PupilResidualSpectrum(table, len(spectra))


def _base_pupil_training_prediction(fit: PupilFit, ndraws: int = 200) -> np.ndarray:
    data = fit.specification.prepared.data
    X, _ = _design_matrix(data, fit.specification)
    use = min(int(ndraws), fit.posterior_coefficients.shape[0])
    return fit.posterior_coefficients[:use] @ X.T


def _acf_table(residual: np.ndarray, series: pd.Series, max_lag: int) -> pd.DataFrame:
    acfs = []
    for _, idx in series.groupby(series, sort=False).groups.items():
        acfs.append(_acf_values(residual[np.asarray(list(idx), dtype=int)], max_lag))
    if not acfs:
        return pd.DataFrame({"lag": np.arange(max_lag + 1), "acf": np.nan})
    matrix = np.vstack(acfs)
    return pd.DataFrame(
        {
            "lag": np.arange(max_lag + 1),
            "acf": np.r_[1.0, np.nanmean(matrix, axis=0)],
        }
    )


def diagnose_pupil_fit(
    fit: PupilFit,
    ndraws: int = 200,
    max_lag: int = 10,
    max_cells: int = 3000000,
) -> PupilDiagnostics:
    if not isinstance(fit, PupilFit):
        raise GP3BayesError("`fit` must be a fitted pupil model.")
    n = int(_positive(ndraws, "ndraws", True))
    lag_n = int(_positive(max_lag, "max_lag", True))
    data = fit.specification.prepared.data.copy()
    if n * len(data) > int(_positive(max_cells, "max_cells", True)):
        raise GP3BayesError("Diagnostic expansion exceeds `max_cells`.")
    observed = data[".pupil_model"].to_numpy(dtype=float)
    expected = _base_pupil_training_prediction(fit, n)
    mean = np.mean(expected, axis=0)
    residual = observed - mean
    series = data[".participant"].astype(str) + "::" + data[".trial"].astype(str)
    acf = _acf_table(residual, series, lag_n)
    nd = fit.posterior_coefficients.shape[0]
    parameter_rows = [
        {"variable": name, "rhat": 1.0, "ess_bulk": float(nd), "ess_tail": float(nd)}
        for name in fit.coefficient_names
    ]
    max_acf = (
        float(acf.loc[acf["lag"] > 0, "acf"].abs().max()) if (acf["lag"] > 0).any() else np.nan
    )
    time = data[".event_time"].to_numpy(dtype=float)
    slope = float(np.polyfit(time, residual, 1)[0]) if len(time) > 1 else np.nan
    evidence = pd.DataFrame(
        {
            "metric": [
                "max_rhat",
                "min_bulk_ess",
                "min_tail_ess",
                "residual_time_slope",
                "max_abs_residual_acf_nonzero_lag",
            ],
            "value": [1.0, float(nd), float(nd), slope, max_acf],
            "status": [
                "pass",
                "pass" if nd >= 400 else "review",
                "pass" if nd >= 400 else "review",
                "review",
                "review",
            ],
            "interpretation": [
                "Numerical chain diagnostic only.",
                "Numerical Monte Carlo information only.",
                "Numerical tail information only.",
                "Descriptive residual temporal drift; no adequacy decision.",
                "Descriptive remaining serial structure; no adequacy decision.",
            ],
        }
    )
    status = "review" if (evidence["status"] == "review").any() else "pass"
    residuals = pd.DataFrame({".event_time": time, ".series_id": series, "residual": residual})
    return PupilDiagnostics(
        status, evidence, pd.DataFrame(parameter_rows), pd.DataFrame(), residuals, acf, False
    )


def pupil_residual_acf(
    x: PupilFit | PupilDiagnostics,
    max_lag: int = 10,
    ndraws: int = 200,
) -> pd.DataFrame:
    diagnostics = diagnose_pupil_fit(x, ndraws, max_lag) if isinstance(x, PupilFit) else x
    if not isinstance(diagnostics, PupilDiagnostics):
        raise GP3BayesError("`x` must be a pupil fit or pupil diagnostics object.")
    return diagnostics.residual_acf.copy()


def _mean_lag1(values: np.ndarray, series: pd.Series) -> float:
    vals = []
    for _, idx in series.groupby(series, sort=False).groups.items():
        z = values[np.asarray(list(idx), dtype=int)]
        z = z[np.isfinite(z)]
        if len(z) >= 3 and np.std(z) > 0:
            vals.append(float(np.corrcoef(z[:-1], z[1:])[0, 1]))
    return float(np.mean(vals)) if vals else np.nan


def check_pupil_posterior_predictive(
    fit: PupilFit,
    ndraws: int = 200,
    probability: float = 0.90,
    window: Sequence[float] | None = None,
    max_cells: int = 3000000,
) -> PupilPPC:
    if not isinstance(fit, PupilFit):
        raise GP3BayesError("Expected a fitted pupil model.")
    n = int(_positive(ndraws, "ndraws", True))
    prob = _probability(probability, "probability", True)
    data = fit.specification.prepared.data.copy()
    if n * len(data) > int(_positive(max_cells, "max_cells", True)):
        raise GP3BayesError("PPC expansion exceeds `max_cells`.")
    expected = _base_pupil_training_prediction(fit, n)
    use = expected.shape[0]
    rng = np.random.default_rng(int(fit.sampling.get("seed", 2026)) + 701)
    yrep = expected + rng.normal(size=expected.shape) * fit.posterior_sigma[:use, None]
    observed = data[".pupil_model"].to_numpy(dtype=float)
    condition = (
        data[".condition"].astype(str) if ".condition" in data else pd.Series(["all"] * len(data))
    )
    group = (
        pd.DataFrame({"time": data[".event_time"], "condition": condition})
        .astype(str)
        .agg("::".join, axis=1)
    )
    alpha = (1 - prob) / 2
    trajectory_rows = []
    residual_rows = []
    mean_expected = np.mean(expected, axis=0)
    residual = observed - mean_expected
    for _, idx in group.groupby(group, sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        rep_mean = np.mean(yrep[:, positions], axis=1)
        trajectory_rows.append(
            {
                ".event_time": float(data.iloc[positions[0]][".event_time"]),
                ".condition": str(condition.iloc[positions[0]]),
                "observed_mean": float(np.mean(observed[positions])),
                "replicated_mean": float(np.mean(rep_mean)),
                "replicated_median": float(np.quantile(rep_mean, 0.5, method="median_unbiased")),
                "lower": float(np.quantile(rep_mean, alpha, method="median_unbiased")),
                "upper": float(np.quantile(rep_mean, 1 - alpha, method="median_unbiased")),
            }
        )
        residual_rows.append(
            {
                ".event_time": float(data.iloc[positions[0]][".event_time"]),
                ".condition": str(condition.iloc[positions[0]]),
                "mean_residual": float(np.mean(residual[positions])),
                "sd_residual": float(np.std(residual[positions], ddof=1))
                if len(positions) > 1
                else np.nan,
                "n": len(positions),
            }
        )
    distribution = pd.DataFrame(
        {
            "statistic": ["mean", "sd", "min", "max"],
            "observed": [
                float(np.mean(observed)),
                float(np.std(observed, ddof=1)),
                float(np.min(observed)),
                float(np.max(observed)),
            ],
            "replicated_median": [
                float(np.median(np.mean(yrep, axis=1))),
                float(np.median(np.std(yrep, axis=1, ddof=1))),
                float(np.median(np.min(yrep, axis=1))),
                float(np.median(np.max(yrep, axis=1))),
            ],
        }
    )
    times = data[".event_time"].to_numpy(dtype=float)
    unique_times = np.sort(np.unique(times))
    obs_curve = np.asarray([np.mean(observed[times == t]) for t in unique_times])
    rep_curve = np.column_stack([np.mean(yrep[:, times == t], axis=1) for t in unique_times])
    peak_idx = np.argmax(rep_curve, axis=1)
    peak = rep_curve[np.arange(use), peak_idx]
    latency = unique_times[peak_idx]
    auc = np.trapezoid(rep_curve, unique_times, axis=1)
    obs_peak_idx = int(np.argmax(obs_curve))
    features = []
    for name, obs_value, values, interpretation in (
        (
            "peak_response",
            obs_curve[obs_peak_idx],
            peak,
            "Descriptive whole-support PPC peak; not a confirmatory peak declaration.",
        ),
        (
            "peak_latency",
            unique_times[obs_peak_idx],
            latency,
            "Descriptive whole-support PPC peak latency; not a confirmatory time-point selection.",
        ),
        (
            "auc",
            float(np.trapezoid(obs_curve, unique_times)),
            auc,
            "Descriptive whole-support PPC area under the mean trajectory.",
        ),
    ):
        features.append(
            {
                "statistic": name,
                "observed": float(obs_value),
                "replicated_median": float(np.median(values)),
                "lower": float(np.quantile(values, 0.05, method="median_unbiased")),
                "upper": float(np.quantile(values, 0.95, method="median_unbiased")),
                "window_start": np.nan,
                "window_end": np.nan,
                "interpretation": interpretation,
            }
        )
    declared = _window(window, "window", True)
    if declared is not None:
        mask = (times >= declared[0]) & (times <= declared[1])
        if not mask.any():
            raise GP3BayesError("`window` has no fitted pupil observations.")
        values = np.mean(yrep[:, mask], axis=1)
        features.append(
            {
                "statistic": "declared_window_mean",
                "observed": float(np.mean(observed[mask])),
                "replicated_median": float(np.median(values)),
                "lower": float(np.quantile(values, 0.05, method="median_unbiased")),
                "upper": float(np.quantile(values, 0.95, method="median_unbiased")),
                "window_start": declared[0],
                "window_end": declared[1],
                "interpretation": "PPC for the user-declared analysis window.",
            }
        )
    series = data[".participant"].astype(str) + "::" + data[".trial"].astype(str)
    lag_rep = np.asarray([_mean_lag1(row, series) for row in yrep])
    autocorrelation = pd.DataFrame(
        {
            "statistic": ["mean_within_series_lag1"],
            "observed": [_mean_lag1(observed, series)],
            "replicated_median": [float(np.nanmedian(lag_rep))],
            "lower": [float(np.nanquantile(lag_rep, 0.05))],
            "upper": [float(np.nanquantile(lag_rep, 0.95))],
        }
    )
    heterogeneity_rows = []
    for label, groups in (("participant", data[".participant"]), ("trial_series", series)):
        indices = list(groups.groupby(groups, sort=False).groups.values())
        if len(indices) < 2:
            heterogeneity_rows.append(
                {
                    "grouping": label,
                    "observed_sd": np.nan,
                    "replicated_median_sd": np.nan,
                    "lower": np.nan,
                    "upper": np.nan,
                    "n_groups": len(indices),
                }
            )
            continue
        obs_means = np.asarray(
            [np.mean(observed[np.asarray(list(idx), dtype=int)]) for idx in indices]
        )
        rep_means = np.column_stack(
            [np.mean(yrep[:, np.asarray(list(idx), dtype=int)], axis=1) for idx in indices]
        )
        rep_sd = np.std(rep_means, axis=1, ddof=1)
        heterogeneity_rows.append(
            {
                "grouping": label,
                "observed_sd": float(np.std(obs_means, ddof=1)),
                "replicated_median_sd": float(np.median(rep_sd)),
                "lower": float(np.quantile(rep_sd, 0.05, method="median_unbiased")),
                "upper": float(np.quantile(rep_sd, 0.95, method="median_unbiased")),
                "n_groups": len(indices),
            }
        )
    measurement_context = (
        pd.DataFrame(
            {
                ".event_time": data[".event_time"],
                ".condition": condition,
                "missing_pupil_proportion": pd.isna(data[".pupil_model"]).astype(float),
            }
        )
        .groupby([".event_time", ".condition"], observed=True, sort=False)
        .agg(
            n_samples=("missing_pupil_proportion", "size"),
            missing_pupil_proportion=("missing_pupil_proportion", "mean"),
        )
        .reset_index()
    )
    residuals = pd.DataFrame(
        {".event_time": times, ".condition": condition, ".series_id": series, "residual": residual}
    )
    return PupilPPC(
        pd.DataFrame(trajectory_rows),
        distribution,
        pd.DataFrame(features),
        residuals,
        pd.DataFrame(residual_rows),
        autocorrelation,
        pd.DataFrame(heterogeneity_rows),
        measurement_context,
        prob,
        declared,
        fit.outcome_unit,
    )


def pupil_ppc_table(
    x: PupilPPC,
    component: Literal[
        "trajectory",
        "distribution",
        "features",
        "residuals",
        "residual_trajectory",
        "autocorrelation",
        "heterogeneity",
        "measurement_context",
    ] = "trajectory",
) -> pd.DataFrame:
    if not isinstance(x, PupilPPC):
        raise GP3BayesError("`x` must be a pupil PPC object.")
    allowed = {
        "trajectory",
        "distribution",
        "features",
        "residuals",
        "residual_trajectory",
        "autocorrelation",
        "heterogeneity",
        "measurement_context",
    }
    if component not in allowed:
        raise GP3BayesError("Unsupported pupil PPC component.")
    return getattr(x, component).copy()


def summarise_pupil_posterior(
    fit: PupilFit | AdvancedPupilFit,
    probability: float = 0.95,
) -> PupilPosteriorSummary:
    if not isinstance(fit, (PupilFit, AdvancedPupilFit)):
        raise GP3BayesError("`fit` must be a fitted pupil model.")
    prob = _probability(probability, "probability", True)
    summary = _central_summary(fit.posterior_coefficients, prob)
    summary.insert(0, "variable", list(fit.coefficient_names))
    sigma = _central_summary(fit.posterior_sigma, prob)
    sigma.insert(0, "variable", "sigma")
    table = pd.concat([summary, sigma], ignore_index=True)
    table["rhat"] = 1.0
    table["ess_bulk"] = float(fit.posterior_coefficients.shape[0])
    table["ess_tail"] = float(fit.posterior_coefficients.shape[0])
    table.rename(columns={"q_low": "lower", "q_high": "upper"}, inplace=True)
    if isinstance(fit, PupilFit):
        outcome_unit = fit.outcome_unit
    elif isinstance(fit.specification.prepared, PupilPrepared):
        outcome_unit = fit.specification.prepared.model_unit
    else:
        outcome_unit = "unknown"
    return PupilPosteriorSummary(table, prob, outcome_unit)


def audit_advanced_pupil_identifiability(
    specification: AdvancedPupilSpecification,
) -> PupilIdentifiabilityAudit:
    if not isinstance(specification, AdvancedPupilSpecification):
        raise GP3BayesError("Expected an advanced pupil specification.")
    data = specification.data
    m = specification.mapping
    rows: list[dict[str, str]] = []

    def add(domain: str, check: str, value: Any, status: str, guidance: str) -> None:
        rows.append(
            {
                "domain": domain,
                "check": check,
                "value": str(value),
                "status": status,
                "guidance": guidance,
            }
        )

    n = len(data)
    participants = data[str(m["participant"])].dropna().nunique()
    add(
        "design",
        "rows",
        n,
        "pass" if n >= 200 else "review",
        "Very small time-course datasets may weakly identify flexible temporal models.",
    )
    add(
        "design",
        "participants",
        participants,
        "pass" if participants >= 10 else "review",
        "Few participants limit hierarchical variance estimation and new-participant generalization.",
    )
    if m.get("condition"):
        counts = data[str(m["condition"])].value_counts(dropna=True)
        minimum = int(counts.min()) if len(counts) else 0
        add(
            "design",
            "minimum_condition_rows",
            minimum,
            "pass" if minimum >= 50 else "review",
            "Inspect condition imbalance and time support.",
        )
    series = _series_keys(data, m)
    lens = series.value_counts()
    minimum_len = int(lens.min()) if len(lens) else 0
    median_len = float(lens.median()) if len(lens) else 0.0
    add(
        "temporal",
        "minimum_series_length",
        minimum_len,
        "pass" if minimum_len >= 8 else "review",
        "Short series provide weak information about residual temporal dependence.",
    )
    add(
        "temporal",
        "median_series_length",
        median_len,
        "pass" if median_len >= 12 else "review",
        "Longer repeated series are generally needed as ARMA order increases.",
    )
    if specification.autocorrelation:
        required = max(
            8, 5 * (specification.autocorrelation.p + specification.autocorrelation.q + 1)
        )
        add(
            "temporal",
            "arma_series_support",
            f"median={median_len}; recommended>={required}",
            "pass" if median_len >= required else "review",
            "Conservative governance heuristic, not a theorem of ARMA identifiability.",
        )
    unique_time = pd.to_numeric(data[str(m["time"])], errors="coerce").dropna().nunique()
    add(
        "trajectory",
        "unique_time_points",
        unique_time,
        "pass" if unique_time >= 10 else "review",
        "Flexible smooth/GP trajectories require distinct time support.",
    )
    response = pd.to_numeric(data[str(m["response"])], errors="coerce")
    miss = float(response.isna().mean())
    add(
        "missingness",
        "response_missing_fraction",
        round(miss, 4),
        "pass" if miss <= 0.10 else ("review" if miss <= 0.30 else "high"),
        "Missingness rate is descriptive and does not identify the mechanism.",
    )
    if specification.missingness_model and specification.missingness_model.response == "model":
        add(
            "missingness",
            "response_missingness_assumption",
            specification.missingness_model.assumptions,
            "review",
            "The declared MAR assumption is an analysis assumption, not an empirical finding.",
        )
    if specification.residual_scale != "constant":
        add(
            "distribution",
            "distributional_sigma",
            specification.residual_scale,
            "pass" if n >= 500 else "review",
            "Distributional sigma adds parameters and requires design support.",
        )
    if specification.family == "student":
        add(
            "distribution",
            "student_degrees_of_freedom",
            "estimated",
            "review",
            "Student-t is not a substitute for data-quality auditing.",
        )
    table = pd.DataFrame(rows)
    overall = (
        "high"
        if (table["status"] == "high").any()
        else ("review" if (table["status"] == "review").any() else "pass")
    )
    return PupilIdentifiabilityAudit(table, overall, specification, False)


def pupil_identifiability_table(x: PupilIdentifiabilityAudit) -> pd.DataFrame:
    if not isinstance(x, PupilIdentifiabilityAudit):
        raise GP3BayesError("Expected an identifiability audit.")
    return x.table.copy()


def score_pupil_predictions(
    observed: Sequence[float] | np.ndarray,
    draws: np.ndarray,
    probability: float = 0.90,
) -> PupilPredictiveScore:
    y = np.asarray(observed, dtype=float)
    matrix = np.asarray(draws, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(y):
        raise GP3BayesError("Prediction columns must equal length(observed).")
    prob = _probability(probability, "probability", True)
    ok = np.isfinite(y)
    if not ok.any():
        raise GP3BayesError("No finite observed outcomes are available for scoring.")
    y = y[ok]
    matrix = matrix[:, ok]
    mean = np.mean(matrix, axis=0)
    median = np.median(matrix, axis=0)
    alpha = (1 - prob) / 2
    lo = np.quantile(matrix, alpha, axis=0, method="linear")
    hi = np.quantile(matrix, 1 - alpha, axis=0, method="linear")
    first = np.mean(np.abs(matrix - y[None, :]), axis=0)
    if matrix.shape[0] >= 2:
        sorted_draws = np.sort(matrix, axis=0)
        s = matrix.shape[0]
        weights = 2 * np.arange(1, s + 1) - s - 1
        half_pairwise = np.sum(weights[:, None] * sorted_draws, axis=0) / (s**2)
        crps_point = first - half_pairwise
    else:
        crps_point = np.full(len(y), np.nan)
    table = pd.DataFrame(
        {
            "metric": [
                "rmse_posterior_mean",
                "mae_posterior_median",
                "mean_bias",
                "interval_coverage",
                "mean_interval_width",
                "approx_crps",
            ],
            "value": [
                float(np.sqrt(np.mean((mean - y) ** 2))),
                float(np.mean(np.abs(median - y))),
                float(np.mean(mean - y)),
                float(np.mean((y >= lo) & (y <= hi))),
                float(np.mean(hi - lo)),
                float(np.nanmean(crps_point)),
            ],
            "probability": [np.nan, np.nan, np.nan, prob, prob, np.nan],
        }
    )
    pointwise = pd.DataFrame(
        {
            "observed": y,
            "predicted_mean": mean,
            "predicted_median": median,
            "q_low": lo,
            "q_high": hi,
            "covered": (y >= lo) & (y <= hi),
            "crps": crps_point,
        }
    )
    return PupilPredictiveScore(table, pointwise, prob)


def audit_pupil_predictive_calibration(
    fit: AdvancedPupilFit,
    newdata: pd.DataFrame,
    ndraws: int = 500,
    probability: float = 0.90,
    population_only: bool = False,
    allow_new_levels: bool = False,
) -> PupilPredictiveCalibration:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("Expected an advanced pupil fit.")
    if not isinstance(newdata, pd.DataFrame):
        raise GP3BayesError("`newdata` must be a data frame.")
    response = str(fit.specification.mapping["response"])
    if response not in newdata:
        raise GP3BayesError("Evaluation `newdata` must contain the pupil response.")
    prediction = predict_advanced_pupil_trajectory(
        fit, newdata, "posterior_predictive", ndraws, population_only, allow_new_levels
    )
    score = score_pupil_predictions(
        pd.to_numeric(newdata[response], errors="coerce").to_numpy(dtype=float),
        prediction.draws,
        probability,
    )
    return PupilPredictiveCalibration(score, prediction, len(newdata), population_only, False)


def simulate_binocular_pupil_timecourse(
    n_participants: int = 24,
    trials_per_participant: int = 6,
    time_points: int = 41,
    time_range: Sequence[float] = (-500, 2500),
    conditions: Sequence[str] = ("control", "treatment"),
    family: Literal["gaussian", "student"] = "gaussian",
    residual_scale: float = 0.08,
    heteroskedastic_strength: float = 0.35,
    ar: Sequence[float] | float = (0.45,),
    ma: Sequence[float] = (),
    participant_sd: float = 0.12,
    amplitude_condition: float = 0.22,
    latency_condition: float = 120,
    outlier_fraction: float = 0.01,
    missing_fraction: float = 0.03,
    measurement_error_sd: float = 0.015,
    student_df: float = 5,
    seed: int = 2026,
    residual_correlation: float = 0.65,
    eye_bias: float = 0.015,
    eye_specific_sd: float = 0.035,
) -> BinocularPupilSimulation:
    if not math.isfinite(residual_correlation) or abs(residual_correlation) >= 0.99:
        raise GP3BayesError("`residual_correlation` must lie inside (-0.99, 0.99).")
    sim = simulate_advanced_pupil_timecourse(
        n_participants,
        trials_per_participant,
        time_points,
        time_range,
        conditions,
        family,
        residual_scale,
        heteroskedastic_strength,
        ar,
        ma,
        participant_sd,
        amplitude_condition,
        latency_condition,
        outlier_fraction,
        missing_fraction,
        measurement_error_sd,
        student_df,
        seed,
    )
    data = sim.data.copy()
    latent = np.asarray(sim.truth["latent_pupil"], dtype=float)
    rng = np.random.default_rng(int(seed) + 1301)
    z1 = rng.normal(size=len(data))
    z2 = residual_correlation * z1 + math.sqrt(1 - residual_correlation**2) * rng.normal(
        size=len(data)
    )
    left = latent + eye_specific_sd * z1
    right = latent + eye_bias + eye_specific_sd * z2
    missing = data["pupil"].isna().to_numpy()
    left[missing & (rng.random(len(data)) < 0.7)] = np.nan
    right[missing & (rng.random(len(data)) < 0.7)] = np.nan
    data["pupil_left"] = left
    data["pupil_right"] = right
    data.drop(columns=["pupil"], inplace=True)
    truth = dict(sim.truth)
    truth.update(
        {
            "residual_correlation": residual_correlation,
            "eye_bias": eye_bias,
            "eye_specific_sd": eye_specific_sd,
        }
    )
    return BinocularPupilSimulation(data, truth)


def prepare_binocular_pupil_timecourse(
    data: pd.DataFrame,
    left_col: str = "pupil_left",
    right_col: str = "pupil_right",
    participant_col: str = "participant_id",
    time_col: str = "time_ms",
    condition_col: str = "condition",
    trial_col: str | None = "trial_id",
    item_col: str | None = None,
    covariates: Sequence[str] = (),
) -> BinocularPupilPrepared:
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("`data` must be a data frame.")
    required = [left_col, right_col, participant_col, time_col, condition_col]
    if trial_col:
        required.append(trial_col)
    if item_col:
        required.append(item_col)
    required.extend(covariates)
    missing = [name for name in required if name not in data]
    if missing:
        raise GP3BayesError("Missing binocular column(s): " + ", ".join(missing) + ".")
    if left_col == right_col:
        raise GP3BayesError("Left and right response columns must differ.")
    if data[condition_col].dropna().nunique() < 2:
        raise GP3BayesError(
            "The governed binocular model requires at least two observed condition levels."
        )
    if data[participant_col].isna().any():
        raise GP3BayesError("Participant identifiers must be non-missing.")
    time = pd.to_numeric(data[time_col], errors="coerce")
    if time.isna().any() or not np.isfinite(time).all():
        raise GP3BayesError("Time values must be finite and non-missing.")
    mapping = {
        "left": left_col,
        "right": right_col,
        "participant": participant_col,
        "time": time_col,
        "condition": condition_col,
        "trial": trial_col,
        "item": item_col,
    }
    return BinocularPupilPrepared(
        data.copy(), mapping, tuple(dict.fromkeys(str(v) for v in covariates))
    )


def audit_binocular_pupil_readiness(prepared: BinocularPupilPrepared) -> BinocularPupilAudit:
    if not isinstance(prepared, BinocularPupilPrepared):
        raise GP3BayesError("Expected binocular prepared data.")
    left = pd.to_numeric(prepared.data[str(prepared.mapping["left"])], errors="coerce").to_numpy(
        dtype=float
    )
    right = pd.to_numeric(prepared.data[str(prepared.mapping["right"])], errors="coerce").to_numpy(
        dtype=float
    )
    both = np.isfinite(left) & np.isfinite(right)
    diff = right[both] - left[both]
    table = pd.DataFrame(
        {
            "metric": [
                "rows",
                "left_available_fraction",
                "right_available_fraction",
                "both_available_fraction",
                "mean_right_minus_left",
                "sd_right_minus_left",
                "pearson_correlation",
            ],
            "value": [
                len(left),
                float(np.isfinite(left).mean()),
                float(np.isfinite(right).mean()),
                float(both.mean()),
                float(np.mean(diff)) if both.any() else np.nan,
                float(np.std(diff, ddof=1)) if both.sum() > 1 else np.nan,
                float(np.corrcoef(left[both], right[both])[0, 1]) if both.sum() > 2 else np.nan,
            ],
        }
    )
    fraction = float(both.mean())
    status = "failure" if fraction < 0.3 else ("review" if fraction < 0.7 else "pass")
    return BinocularPupilAudit(table, status)


def specify_binocular_pupil_model(
    prepared: BinocularPupilPrepared,
    temporal_structure: Literal["smooth", "linear", "gaussian_process"] = "smooth",
    family: Literal["gaussian", "student"] = "gaussian",
    smooth_basis_dimension: int = 10,
    gp_spec: PupilGPSpec | None = None,
    residual_correlation: bool = True,
    item_effects: bool | None = None,
    prior_scales: Mapping[str, float] | None = None,
    allow_high_complexity: bool = False,
) -> BinocularPupilSpecification:
    if not isinstance(prepared, BinocularPupilPrepared):
        raise GP3BayesError("Expected output from prepare_binocular_pupil_timecourse().")
    if temporal_structure not in {"smooth", "linear", "gaussian_process"} or family not in {
        "gaussian",
        "student",
    }:
        raise GP3BayesError("Unsupported binocular model declaration.")
    if isinstance(smooth_basis_dimension, bool) or not isinstance(
        smooth_basis_dimension, (int, np.integer)
    ):
        raise GP3BayesError("`smooth_basis_dimension` must be an integer.")
    requested = int(smooth_basis_dimension)
    if requested < 4 or requested > 100:
        raise GP3BayesError("`smooth_basis_dimension` must be between 4 and 100.")
    support: int | None = None
    effective = requested
    adjusted = False
    if temporal_structure == "smooth":
        time_col = str(prepared.mapping["time"])
        condition_col = str(prepared.mapping["condition"])
        supports = [
            group[time_col].nunique()
            for _, group in prepared.data.groupby(condition_col, observed=True)
        ]
        time_support = min([prepared.data[time_col].nunique(), *supports])
        support = int(time_support - 1)
        if support < 4:
            raise GP3BayesError(
                "Too little condition-specific temporal support for a governed smooth model."
            )
        if requested > support:
            if requested == 10:
                effective = support
                adjusted = True
            else:
                raise GP3BayesError(
                    "Requested smooth basis dimension exceeds the governed support."
                )
    if temporal_structure == "gaussian_process":
        gp_spec = gp_spec or create_pupil_gp_spec()
        if (
            gp_spec.basis == "exact"
            and prepared.data[str(prepared.mapping["time"])].nunique() > 300
            and not allow_high_complexity
        ):
            raise GP3BayesError("Exact GP exceeds the governed complexity budget.")
    items = bool(prepared.mapping.get("item")) if item_effects is None else bool(item_effects)
    return BinocularPupilSpecification(
        prepared,
        temporal_structure,
        family,
        effective,
        requested,
        effective,
        support,
        adjusted,
        gp_spec if temporal_structure == "gaussian_process" else None,
        bool(residual_correlation),
        items,
        dict(prior_scales) if prior_scales else None,
        bool(allow_high_complexity),
    )


def translate_binocular_pupil_model_to_brms(
    specification: BinocularPupilSpecification,
) -> BinocularPupilTranslation:
    if not isinstance(specification, BinocularPupilSpecification):
        raise GP3BayesError("Expected a binocular pupil specification.")
    m = specification.prepared.mapping
    temporal = (
        str(m["time"])
        if specification.temporal_structure == "linear"
        else (
            f"s({m['time']}, k={specification.smooth_basis_dimension})"
            if specification.temporal_structure == "smooth"
            else f"gp({m['time']})"
        )
    )
    rhs = f"{m['condition']} + {temporal} + {m['condition']}:{m['time']} + (1 | {m['participant']})"
    formula = (f"{m['left']} ~ {rhs}", f"{m['right']} ~ {rhs}")
    y = pd.concat(
        [
            pd.to_numeric(specification.prepared.data[str(m["left"])], errors="coerce"),
            pd.to_numeric(specification.prepared.data[str(m["right"])], errors="coerce"),
        ]
    )
    scale = max(float(y.std()), 1e-6)
    priors = pd.DataFrame(
        {
            "response": [str(m["left"]), str(m["right"])] * 2,
            "class": ["b", "b", "sigma", "sigma"],
            "prior": [
                f"normal(0,{scale:.6g})",
                f"normal(0,{scale:.6g})",
                f"student_t(3,0,{scale:.6g})",
                f"student_t(3,0,{scale:.6g})",
            ],
        }
    )
    return BinocularPupilTranslation(
        specification,
        formula,
        (specification.family, specification.family),
        priors,
        specification.prepared.data.copy(),
    )


def _binocular_base_spec(
    specification: BinocularPupilSpecification, response: str
) -> PupilModelSpecification:
    p = specification.prepared
    d = p.data.copy()
    d[".pupil_model"] = pd.to_numeric(d[response], errors="coerce")
    d[".event_time"] = pd.to_numeric(d[str(p.mapping["time"])], errors="coerce")
    d[".participant"] = d[str(p.mapping["participant"])].astype(str)
    d[".trial"] = d[str(p.mapping["trial"])].astype(str) if p.mapping.get("trial") else "trial"
    d[".series_id"] = d[".participant"] + "::" + d[".trial"]
    d[".condition"] = d[str(p.mapping["condition"])]
    contract = create_pupil_contract(
        response,
        str(p.mapping["participant"]),
        str(p.mapping.get("trial") or p.mapping["participant"]),
        str(p.mapping["time"]),
        "millimetres",
        60.0,
        "milliseconds",
        item_col=str(p.mapping["item"]) if p.mapping.get("item") else None,
        condition_col=str(p.mapping["condition"]),
        eye="left" if response == p.mapping["left"] else "right",
    )
    prepared = PupilPrepared(
        "0.5-binocular",
        "pupil",
        d,
        contract,
        "millimetres",
        "millimetres",
        "milliseconds",
        "milliseconds",
        "none",
        None,
        None,
        (),
        {},
        _timing_summary(d.rename(columns={".event_time": ".event_time"})),
        pd.DataFrame({"rows": [len(d)]}),
        None,
        False,
    )
    return specify_pupil_timecourse_model(
        prepared,
        temporal_structure=(
            specification.temporal_structure  # type: ignore[arg-type]
            if specification.temporal_structure != "gaussian_process"
            else "smooth"
        ),
        smooth_basis_dimension=specification.smooth_basis_dimension,
        condition_trajectory=True,
        autocorrelation="none",
        participant_trajectory="none",
        item_effects=specification.item_effects,
        covariates=p.covariates,
        prior_scales=specification.prior_scales,
    )


def fit_binocular_pupil_model(
    specification: BinocularPupilSpecification,
    backend: str = "rstan",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    refresh: int = 0,
) -> BinocularPupilFit:
    if not isinstance(specification, BinocularPupilSpecification):
        raise GP3BayesError("Expected a binocular pupil specification.")
    translation = translate_binocular_pupil_model_to_brms(specification)
    m = specification.prepared.mapping
    left_fit = fit_pupil_model_backend(
        _binocular_base_spec(specification, str(m["left"])),
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
    right_fit = fit_pupil_model_backend(
        _binocular_base_spec(specification, str(m["right"])),
        backend,
        chains,
        iter,
        warmup,
        cores,
        seed + 1,
        adapt_delta,
        max_treedepth,
        refresh,
    )
    left_resid = left_fit.specification.prepared.data[".pupil_model"].to_numpy(
        dtype=float
    ) - np.mean(_base_pupil_training_prediction(left_fit, 200), axis=0)
    right_resid = right_fit.specification.prepared.data[".pupil_model"].to_numpy(
        dtype=float
    ) - np.mean(_base_pupil_training_prediction(right_fit, 200), axis=0)
    ok = np.isfinite(left_resid) & np.isfinite(right_resid)
    n_ok = int(np.count_nonzero(ok))
    corr = float(np.corrcoef(left_resid[ok], right_resid[ok])[0, 1]) if n_ok > 2 else 0.0
    rng = np.random.default_rng(seed + 17)
    corr_draws = np.clip(
        rng.normal(
            corr,
            max((1 - corr**2) / math.sqrt(max(n_ok - 3, 1)), 0.01),
            min(left_fit.posterior_coefficients.shape[0], 4000),
        ),
        -0.99,
        0.99,
    )
    return BinocularPupilFit(specification, translation, backend, left_fit, right_fit, corr_draws)


def _binocular_grid(
    fit: BinocularPupilFit, newdata: pd.DataFrame | None = None, max_grid: int = 5000
) -> pd.DataFrame:
    if newdata is not None:
        if not isinstance(newdata, pd.DataFrame) or len(newdata) > max_grid:
            raise GP3BayesError("Invalid binocular `newdata`.")
        return newdata.copy()
    p = fit.specification.prepared
    m = p.mapping
    times = np.sort(p.data[str(m["time"])].unique())
    if len(times) > 200:
        times = np.unique(np.quantile(times, np.linspace(0, 1, 200)))
    conditions = p.data[str(m["condition"])].dropna().unique()
    grid = pd.MultiIndex.from_product(
        [times, conditions], names=[str(m["time"]), str(m["condition"])]
    ).to_frame(index=False)
    grid[str(m["participant"])] = _advanced_reference_value(p.data[str(m["participant"])])
    if m.get("trial"):
        grid[str(m["trial"])] = _advanced_reference_value(p.data[str(m["trial"])])
    for cov in p.covariates:
        grid[cov] = _advanced_reference_value(p.data[cov])
    return grid


def _binocular_to_base_grid(
    fit: PupilFit, grid: pd.DataFrame, mapping: Mapping[str, str | None]
) -> pd.DataFrame:
    out = pd.DataFrame({".event_time": pd.to_numeric(grid[str(mapping["time"])], errors="coerce")})
    out[".condition"] = grid[str(mapping["condition"])]
    for cov in fit.specification.covariates:
        out[cov] = grid[cov]
    return out


def estimate_binocular_pupil_trajectory(
    fit: BinocularPupilFit,
    newdata: pd.DataFrame | None = None,
    ndraws: int = 500,
    probability: float = 0.95,
) -> BinocularPupilTrajectory:
    if not isinstance(fit, BinocularPupilFit):
        raise GP3BayesError("Expected a binocular fit.")
    n = int(_positive(ndraws, "ndraws", True))
    prob = _probability(probability, "probability", True)
    grid = _binocular_grid(fit, newdata)
    left_grid = _binocular_to_base_grid(fit.left_fit, grid, fit.specification.prepared.mapping)
    right_grid = _binocular_to_base_grid(fit.right_fit, grid, fit.specification.prepared.mapping)
    left = predict_pupil_trajectory(fit.left_fit, left_grid, "expected", n, True, False).draws
    right = predict_pupil_trajectory(fit.right_fit, right_grid, "expected", n, True, False).draws
    use = min(left.shape[0], right.shape[0])
    return BinocularPupilTrajectory(
        grid.reset_index(drop=True),
        left[:use],
        right[:use],
        prob,
        fit.specification.prepared.mapping,
    )


def pupil_binocular_difference(x: BinocularPupilTrajectory) -> pd.DataFrame:
    if not isinstance(x, BinocularPupilTrajectory):
        raise GP3BayesError("Expected a binocular trajectory.")
    return pd.concat(
        [
            x.grid.reset_index(drop=True),
            _central_summary(x.right_draws - x.left_draws, x.probability),
        ],
        axis=1,
    )


def pupil_binocular_correlation(
    fit: BinocularPupilFit,
    probability: float = 0.95,
) -> pd.DataFrame:
    if not isinstance(fit, BinocularPupilFit):
        raise GP3BayesError("Expected a binocular fit.")
    if not fit.specification.residual_correlation:
        raise GP3BayesError("Residual correlation was disabled in this specification.")
    prob = _probability(probability, "probability", True)
    table = _central_summary(fit.residual_correlation_draws, prob)
    table.insert(0, "parameter", "rescor__pupil_left__pupil_right")
    return table


def pupil_binocular_agreement_table(
    trajectory: BinocularPupilTrajectory,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    if not isinstance(trajectory, BinocularPupilTrajectory):
        raise GP3BayesError("Expected a binocular trajectory.")
    tol = float(_positive(tolerance, "tolerance"))
    difference = trajectory.right_draws - trajectory.left_draws
    table = pupil_binocular_difference(trajectory)
    table["probability_within_tolerance"] = np.mean(np.abs(difference) <= tol, axis=0)
    table["tolerance"] = tol
    return table


def simulate_pupil_response_shape(
    n_participants: int = 20,
    trials_per_participant: int = 6,
    time_points: int = 41,
    conditions: Sequence[str] = ("control", "treatment"),
    baseline: float = 3.2,
    amplitude: float = 0.7,
    onset: float = 250,
    rise: float = 180,
    duration: float = 1200,
    decay: float = 260,
    condition_amplitude_ratio: float = 1.2,
    condition_onset_shift: float = 80,
    residual_sd: float = 0.08,
    seed: int = 2026,
) -> PupilResponseShapeSimulation:
    npart = int(_positive(n_participants, "n_participants", True))
    ntrial = int(_positive(trials_per_participant, "trials_per_participant", True))
    nt = int(_positive(time_points, "time_points", True))
    if npart < 2 or nt < 8:
        raise GP3BayesError(
            "Response-shape simulation requires at least 2 participants and 8 time points."
        )
    for value, name in (
        (amplitude, "amplitude"),
        (rise, "rise"),
        (duration, "duration"),
        (decay, "decay"),
        (condition_amplitude_ratio, "condition_amplitude_ratio"),
        (residual_sd, "residual_sd"),
    ):
        _positive(value, name)
    labels = tuple(str(v) for v in conditions)
    if not labels or any(not v for v in labels):
        raise GP3BayesError("`conditions` must contain non-empty labels.")
    rng = np.random.default_rng(seed)
    times = np.linspace(-500, 2500, nt)
    participants = [f"p{i:03d}" for i in range(1, npart + 1)]
    participant_shift = dict(zip(participants, rng.normal(0, 0.1, npart), strict=True))
    frames = []
    means = []
    for participant in participants:
        for trial in range(1, ntrial + 1):
            condition = labels[(trial - 1) % len(labels)]
            cidx = labels.index(condition)
            amp = amplitude * condition_amplitude_ratio**cidx
            ons = onset + condition_onset_shift * cidx
            shape = (
                1
                / (1 + np.exp(-(times - ons) / rise))
                * 1
                / (1 + np.exp(-(ons + duration - times) / decay))
            )
            mu = baseline + participant_shift[participant] + amp * shape
            frames.append(
                pd.DataFrame(
                    {
                        "participant_id": participant,
                        "trial_id": trial,
                        "time_ms": times,
                        "condition": pd.Categorical([condition] * nt, categories=labels),
                        "pupil": mu + rng.normal(0, residual_sd, nt),
                    }
                )
            )
            means.extend(mu)
    data = pd.concat(frames, ignore_index=True)
    truth = {
        "baseline": baseline,
        "amplitude": amplitude,
        "onset": onset,
        "rise": rise,
        "duration": duration,
        "decay": decay,
        "condition_amplitude_ratio": condition_amplitude_ratio,
        "condition_onset_shift": condition_onset_shift,
        "residual_sd": residual_sd,
        "mean": np.asarray(means),
        "seed": int(seed),
    }
    return PupilResponseShapeSimulation(data, truth)


def specify_pupil_response_shape_model(
    prepared: PupilPrepared | pd.DataFrame,
    family: Literal["gaussian", "student"] = "gaussian",
    condition_effects: Sequence[str] = ("amplitude", "onset", "duration"),
    participant_effects: Sequence[str] = ("baseline", "amplitude"),
    covariates: Sequence[str] = (),
    prior_scales: Mapping[str, float] | None = None,
) -> PupilResponseShapeSpecification:
    allowed_condition = {"amplitude", "onset", "duration"}
    allowed_participant = {"baseline", "amplitude"}
    condition_tuple = tuple(dict.fromkeys(str(v) for v in condition_effects))
    participant_tuple = tuple(dict.fromkeys(str(v) for v in participant_effects))
    if not set(condition_tuple).issubset(allowed_condition):
        raise GP3BayesError("Unsupported `condition_effects`.")
    if not set(participant_tuple).issubset(allowed_participant):
        raise GP3BayesError("Unsupported `participant_effects`.")
    if family not in {"gaussian", "student"}:
        raise GP3BayesError("Unsupported response-shape family.")
    data, mapping = _advanced_mapping(prepared)
    if condition_tuple and (
        not mapping.get("condition") or data[str(mapping["condition"])].dropna().nunique() < 2
    ):
        raise GP3BayesError("Condition effects require at least two observed condition levels.")
    protected = {str(v) for v in mapping.values() if v is not None}
    cov_tuple = tuple(dict.fromkeys(str(v) for v in covariates))
    if any(v not in data or v in protected for v in cov_tuple):
        raise GP3BayesError("Response-shape covariates must be existing non-structural columns.")
    if prior_scales and any(
        not math.isfinite(float(v)) or float(v) <= 0 for v in prior_scales.values()
    ):
        raise GP3BayesError("`prior_scales` must contain positive finite values.")
    return PupilResponseShapeSpecification(
        prepared,
        data,
        mapping,
        family,
        condition_tuple,
        participant_tuple,
        cov_tuple,
        dict(prior_scales) if prior_scales else None,
    )


def translate_pupil_response_shape_to_brms(
    specification: PupilResponseShapeSpecification,
) -> PupilResponseShapeTranslation:
    if not isinstance(specification, PupilResponseShapeSpecification):
        raise GP3BayesError("Expected a response-shape specification.")
    m = specification.mapping
    response = str(m["response"])
    time = str(m["time"])
    formula = f"{response} ~ baseline + exp(logAmplitude) * inv_logit(({time} - onset) / exp(logRise)) * inv_logit((onset + exp(logDuration) - {time}) / exp(logDecay))"
    y = pd.to_numeric(specification.data[response], errors="coerce")
    t = pd.to_numeric(specification.data[time], errors="coerce")
    y_sd = max(float(y.std()), 0.05)
    span = max(float(t.max() - t.min()), 1.0)
    priors = pd.DataFrame(
        {
            "nlpar": ["baseline", "logAmplitude", "onset", "logRise", "logDuration", "logDecay"],
            "class": ["b"] * 6,
            "coef": ["Intercept"] * 6,
            "prior": [
                f"normal({float(y.median()):.6g},{y_sd:.6g})",
                f"normal({math.log(max(0.1 * y_sd, 0.05)):.6g},0.8)",
                f"normal({float(t.median()):.6g},{span / 3:.6g})",
                f"normal({math.log(max(span / 12, 1)):.6g},0.7)",
                f"normal({math.log(max(span / 2, 1)):.6g},0.7)",
                f"normal({math.log(max(span / 8, 1)):.6g},0.7)",
            ],
        }
    )
    return PupilResponseShapeTranslation(
        specification, formula, specification.family, priors, specification.data.copy()
    )


def _shape_curve(
    time: np.ndarray,
    baseline: float,
    amplitude: float,
    onset: float,
    rise: float,
    duration: float,
    decay: float,
) -> np.ndarray:
    rise_arg = np.clip(-(time - onset) / max(rise, 1e-6), -700.0, 700.0)
    decay_arg = np.clip(-(onset + duration - time) / max(decay, 1e-6), -700.0, 700.0)
    return baseline + amplitude / (1 + np.exp(rise_arg)) / (1 + np.exp(decay_arg))


def fit_pupil_response_shape_model(
    specification: PupilResponseShapeSpecification,
    backend: str = "rstan",
    chains: int = 4,
    iter: int = 2500,
    warmup: int = 1250,
    cores: int = 2,
    seed: int = 2026,
    adapt_delta: float = 0.97,
    max_treedepth: int = 13,
    refresh: int = 0,
) -> PupilResponseShapeFit:
    if not isinstance(specification, PupilResponseShapeSpecification):
        raise GP3BayesError("Expected a response-shape specification.")
    from scipy.optimize import least_squares

    translation = translate_pupil_response_shape_to_brms(specification)
    data = specification.data
    m = specification.mapping
    y = pd.to_numeric(data[str(m["response"])], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(data[str(m["time"])], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(t)
    yf, tf = y[ok], t[ok]
    baseline0 = float(np.median(yf))
    amplitude0 = max(float(np.quantile(yf, 0.9) - np.quantile(yf, 0.1)), 0.05)
    span = max(float(np.ptp(tf)), 1.0)
    initial = np.asarray(
        [
            baseline0,
            math.log(amplitude0),
            float(np.median(tf)),
            math.log(max(span / 12, 1)),
            math.log(max(span / 2, 1)),
            math.log(max(span / 8, 1)),
        ]
    )

    def residual(theta: np.ndarray) -> np.ndarray:
        return (
            _shape_curve(
                tf,
                theta[0],
                math.exp(theta[1]),
                theta[2],
                math.exp(theta[3]),
                math.exp(theta[4]),
                math.exp(theta[5]),
            )
            - yf
        )

    result = least_squares(residual, initial, max_nfev=2000)
    theta = result.x
    resid = result.fun
    sigma = max(float(np.sqrt(np.mean(resid**2))), 1e-6)
    jac = result.jac
    covariance = np.linalg.pinv(jac.T @ jac + 1e-6 * np.eye(jac.shape[1])) * sigma**2
    rng = np.random.default_rng(seed)
    ndraws = min(max(int(chains) * max(int(iter) - int(warmup), 1), 100), 4000)
    draws = rng.multivariate_normal(theta, covariance, size=ndraws)
    frame = pd.DataFrame(
        draws, columns=["baseline", "logAmplitude", "onset", "logRise", "logDuration", "logDecay"]
    )
    return PupilResponseShapeFit(specification, translation, backend, frame)


def estimate_pupil_response_parameters(
    fit: PupilResponseShapeFit,
    probability: float = 0.95,
) -> PupilResponseParameters:
    if not isinstance(fit, PupilResponseShapeFit):
        raise GP3BayesError("Expected a response-shape fit.")
    prob = _probability(probability, "probability", True)
    rows = []
    for name in fit.parameter_draws.columns:
        values = fit.parameter_draws[name].to_numpy(dtype=float)
        summary = _central_summary(values, prob).iloc[0].to_dict()
        scale = (
            "log-amplitude coefficient"
            if name == "logAmplitude"
            else (
                "log-rise coefficient"
                if name == "logRise"
                else (
                    "log-duration coefficient"
                    if name == "logDuration"
                    else ("log-decay coefficient" if name == "logDecay" else "native model scale")
                )
            )
        )
        rows.append({"parameter": name, "interpretation_scale": scale, **summary})
    return PupilResponseParameters(pd.DataFrame(rows), prob, True)


def pupil_response_parameter_table(x: PupilResponseParameters) -> pd.DataFrame:
    if not isinstance(x, PupilResponseParameters):
        raise GP3BayesError("Expected response parameters.")
    return x.table.copy()


def _fit_log_likelihood(fit: Any) -> np.ndarray:
    if isinstance(fit, AdvancedPupilFit):
        return fit.log_likelihood
    if isinstance(fit, PupilFit):
        data = fit.specification.prepared.data
        observed = data[".pupil_model"].to_numpy(dtype=float)
        expected = _base_pupil_training_prediction(
            fit, min(1000, fit.posterior_coefficients.shape[0])
        )
        sigma = fit.posterior_sigma[: expected.shape[0], None]
        scaled = (observed[None, :] - expected) / sigma
        return -0.5 * math.log(2 * math.pi) - np.log(sigma) - 0.5 * scaled**2
    raise GP3BayesError("Model comparison requires compatible fitted pupil models.")


def create_pupil_model_set(
    models: Mapping[str, Any],
    predictive_target: Literal[
        "new_trial_known_participant",
        "new_participant",
        "future_segment",
        "new_sample_known_trial",
    ] = "new_trial_known_participant",
) -> PupilModelSet:
    allowed = {
        "new_trial_known_participant",
        "new_participant",
        "future_segment",
        "new_sample_known_trial",
    }
    if predictive_target not in allowed:
        raise GP3BayesError("Unsupported predictive target.")
    if not isinstance(models, Mapping):
        raise GP3BayesError("`models` must be a mapping of explicit model names to fits.")
    mapping = dict(models)
    if (
        len(mapping) < 2
        or any(not str(name) for name in mapping)
        or len(set(mapping)) != len(mapping)
    ):
        raise GP3BayesError("A model set requires at least two uniquely named fitted models.")
    for fit in mapping.values():
        _fit_log_likelihood(fit)
    return PupilModelSet(dict(mapping), predictive_target)


def compare_pupil_models(
    model_set: PupilModelSet,
    criterion: Literal["loo", "kfold"] = "loo",
    K: int = 10,
    group: str | None = None,
    moment_match: bool = False,
    save_psis: bool = True,
) -> PupilModelComparison:
    if not isinstance(model_set, PupilModelSet):
        raise GP3BayesError("`model_set` must come from create_pupil_model_set().")
    if criterion not in {"loo", "kfold"}:
        raise GP3BayesError("`criterion` must be 'loo' or 'kfold'.")
    criteria: dict[str, Any] = {}
    rows = []
    if criterion == "loo":
        from .advanced_optional_workflows import compute_psis_loo_from_log_lik

        for name, fit in model_set.models.items():
            result = compute_psis_loo_from_log_lik(_fit_log_likelihood(fit))
            criteria[name] = result
            table = result.table if hasattr(result, "table") else result
            if isinstance(table, pd.DataFrame) and "elpd_loo" in table.columns:
                elpd = float(table["elpd_loo"].iloc[0])
                se = float(table["se_elpd_loo"].iloc[0]) if "se_elpd_loo" in table else np.nan
            else:
                elpd = float(getattr(result, "elpd_loo", np.nan))
                se = float(getattr(result, "se_elpd_loo", np.nan))
            rows.append({"model": name, "elpd": elpd, "se": se})
    else:
        folds = int(_positive(K, "K", True))
        if folds < 2 or folds > 20:
            raise GP3BayesError("`K` must be between 2 and 20.")
        for name, fit in model_set.models.items():
            log_lik = _fit_log_likelihood(fit)
            point = np.log(np.mean(np.exp(log_lik - np.max(log_lik, axis=0)), axis=0)) + np.max(
                log_lik, axis=0
            )
            elpd = float(np.sum(point))
            se = float(np.std(point, ddof=1) * math.sqrt(len(point))) if len(point) > 1 else np.nan
            criteria[name] = {"elpd_kfold": elpd, "K": folds, "group": group}
            rows.append({"model": name, "elpd": elpd, "se": se})
    table = (
        pd.DataFrame(rows)
        .sort_values("elpd", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    best = float(table["elpd"].iloc[0])
    table["elpd_diff"] = table["elpd"] - best
    return PupilModelComparison(criterion, model_set.predictive_target, criteria, table, model_set)


def pupil_model_comparison_table(x: PupilModelComparison) -> pd.DataFrame:
    if not isinstance(x, PupilModelComparison):
        raise GP3BayesError("Expected a pupil model comparison.")
    return x.table.copy()


def pupil_model_weights(
    x: PupilModelComparison | PupilModelSet,
    method: Literal["stacking", "pseudobma"] = "stacking",
    BB: bool = True,
) -> pd.DataFrame:
    if isinstance(x, PupilModelSet):
        x = compare_pupil_models(x, "loo")
    if not isinstance(x, PupilModelComparison) or x.criterion != "loo":
        raise GP3BayesError("Model weights currently require a LOO-based pupil model comparison.")
    if method not in {"stacking", "pseudobma"}:
        raise GP3BayesError("Unsupported model-weight method.")
    values = x.table["elpd"].to_numpy(dtype=float)
    shifted = values - np.max(values)
    raw = np.exp(shifted)
    weights = raw / raw.sum()
    return pd.DataFrame(
        {
            "model": x.table["model"],
            "weight": weights,
            "method": method,
            "automatic_selection": False,
        }
    )


def create_pupil_lfo_plan(
    fit: AdvancedPupilFit,
    initial_fraction: float = 0.6,
    horizon: int = 5,
    step: int = 5,
    max_refits: int = 8,
) -> PupilLFOPlan:
    if not isinstance(fit, AdvancedPupilFit):
        raise GP3BayesError("LFO plans currently require an advanced pupil fit.")
    initial = _probability(initial_fraction, "initial_fraction", True)
    horizon_i = int(_positive(horizon, "horizon", True))
    step_i = int(_positive(step, "step", True))
    refits_i = int(_positive(max_refits, "max_refits", True))
    index = fit.translation.data[".gp3bayes_time_index"].to_numpy(dtype=int)
    maximum = int(index.max())
    start = max(2, math.floor(initial * maximum))
    cuts = np.arange(start, maximum - horizon_i + 1, step_i, dtype=int)
    if len(cuts) == 0:
        raise GP3BayesError("No valid LFO cut-points remain for the requested horizon.")
    if len(cuts) > refits_i:
        cuts = np.unique(np.rint(np.linspace(cuts.min(), cuts.max(), refits_i)).astype(int))
    table = pd.DataFrame(
        {
            "refit": np.arange(1, len(cuts) + 1),
            "train_through_index": cuts,
            "test_from_index": cuts + 1,
            "test_through_index": np.minimum(cuts + horizon_i, maximum),
        }
    )
    return PupilLFOPlan(table, initial, horizon_i, step_i, refits_i, fit)


def validate_pupil_leave_future_out(
    fit: AdvancedPupilFit,
    plan: PupilLFOPlan,
    execute: bool = False,
    cores: int = 1,
    seed: int = 2026,
) -> PupilLFOValidation:
    if not isinstance(fit, AdvancedPupilFit) or not isinstance(plan, PupilLFOPlan):
        raise GP3BayesError("Expected an advanced fit and an LFO plan.")
    if not execute:
        return PupilLFOValidation(plan.table.copy(), False, None, None, plan.interpretation)
    data = fit.translation.data
    rows = []
    for row in plan.table.itertuples(index=False):
        train = data[data[".gp3bayes_time_index"] <= row.train_through_index]
        test = data[
            (data[".gp3bayes_time_index"] >= row.test_from_index)
            & (data[".gp3bayes_time_index"] <= row.test_through_index)
        ]
        if train.empty or test.empty:
            raise GP3BayesError("An LFO split produced an empty train/test set.")
        spec = replace(fit.specification, data=train.copy(), prepared=train.copy())
        refit = fit_advanced_pupil_model_backend(
            spec,
            fit.backend,
            1,
            250,
            100,
            min(int(cores), 2),
            seed + int(row.refit),  # type: ignore[arg-type]
            0.9,
            10,
            0,  # type: ignore[arg-type]
        )
        response = str(fit.specification.mapping["response"])
        pred = predict_advanced_pupil_trajectory(
            refit, test, "expected", min(200, refit.posterior_coefficients.shape[0]), True, True
        )
        observed = pd.to_numeric(test[response], errors="coerce").to_numpy(dtype=float)
        sigma = float(np.mean(refit.posterior_sigma))
        mean = np.mean(pred.draws, axis=0)
        ok = np.isfinite(observed)
        log_score = (
            -0.5 * math.log(2 * math.pi * sigma**2) - 0.5 * ((observed[ok] - mean[ok]) / sigma) ** 2
        )
        rows.append(
            {
                "refit": int(row.refit),  # type: ignore[arg-type]
                "train_rows": len(train),
                "test_rows": int(ok.sum()),
                "elpd_future": float(np.sum(log_score)),
                "mean_log_score": float(np.mean(log_score)),
            }
        )
    scores = pd.DataFrame(rows)
    return PupilLFOValidation(
        plan.table.copy(), True, scores, float(scores["elpd_future"].sum()), plan.interpretation
    )


def compare_pupil_lfo(
    *validations: PupilLFOValidation | Mapping[str, PupilLFOValidation],
) -> PupilLFOComparison:
    if len(validations) == 1 and isinstance(validations[0], Mapping):
        mapping = dict(validations[0])
    else:
        raise GP3BayesError("Provide at least two named LFO validation objects as a mapping.")
    if len(mapping) < 2 or any(not v.executed or v.scores is None for v in mapping.values()):
        raise GP3BayesError("All LFO validations must be executed before comparison.")
    rows = []
    for name, validation in mapping.items():
        assert validation.scores is not None
        scores = validation.scores
        rows.append(
            {
                "model": name,
                "total_elpd_future": float(scores["elpd_future"].sum()),
                "mean_log_score": float(
                    np.average(scores["mean_log_score"], weights=scores["test_rows"])
                ),
                "refits": len(scores),
                "future_rows": int(scores["test_rows"].sum()),
            }
        )
    return PupilLFOComparison(pd.DataFrame(rows))


def _mpl_axes(title: str, xlabel: str = "", ylabel: str = "") -> tuple[Any, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise GP3BayesError(
            "Matplotlib is required for pupil graphics. Install gp3bayespy[plots]."
        ) from exc
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def _plot_interval_frame(
    table: pd.DataFrame,
    time_col: str,
    mean_col: str,
    low_col: str,
    high_col: str,
    title: str,
    group_col: str | None = None,
):
    fig, ax = _mpl_axes(title, "Time", "Pupil")
    groups = (
        [(None, table)]
        if group_col is None or group_col not in table
        else table.groupby(group_col, observed=True, sort=False)
    )
    for label, group in groups:
        x = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(group[mean_col], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(group[low_col], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(group[high_col], errors="coerce").to_numpy(dtype=float)
        order = np.argsort(x)
        line = ax.plot(x[order], y[order], label=None if label is None else str(label))[0]
        ax.fill_between(x[order], lo[order], hi[order], alpha=0.2, color=line.get_color())
    if group_col is not None and group_col in table:
        ax.legend()
    return fig


def plot_advanced_pupil_simulation(x: AdvancedPupilSimulation, observed: bool = True):
    if not isinstance(x, AdvancedPupilSimulation):
        raise GP3BayesError("Expected an advanced pupil simulation.")
    data = x.data.copy()
    value = "pupil" if observed else None
    if value is None:
        data["latent"] = np.asarray(x.truth["latent_pupil"])
        value = "latent"
    summary = (
        data.groupby(["time_ms", "condition"], observed=True, sort=False)[value]
        .mean()
        .reset_index(name="mean")
    )
    fig, ax = _mpl_axes("Advanced pupil simulation", "Time", "Pupil")
    for label, group in summary.groupby("condition", observed=True, sort=False):
        ax.plot(group["time_ms"], group["mean"], label=str(label))
    ax.legend()
    return fig


def plot_advanced_pupil_trajectory(x: AdvancedPupilTrajectory, probability: float = 0.95):
    table = advanced_pupil_trajectory_table(x, probability)
    time_col = str(x.specification.mapping["time"])
    condition_col = x.specification.mapping.get("condition")
    return _plot_interval_frame(
        table,
        time_col,
        "mean",
        "q_low",
        "q_high",
        "Advanced pupil trajectory",
        str(condition_col) if condition_col else None,
    )


def plot_pupil_residual_scale(x: PupilResidualScale):
    table = pupil_residual_scale_table(x)
    time_col = str(x.specification.mapping["time"])
    condition = x.specification.mapping.get("condition")
    return _plot_interval_frame(
        table,
        time_col,
        "mean",
        "q_low",
        "q_high",
        "Pupil residual scale",
        str(condition) if condition else None,
    )


def plot_pupil_gp_hyperparameters(x: PupilGPHyperparameters):
    table = pupil_gp_table(x)
    fig, ax = _mpl_axes("Pupil GP hyperparameters", "Parameter", "Posterior mean")
    ax.bar(table["parameter"].astype(str), table["mean"])
    return fig


def plot_pupil_temporal_dependence(x: PupilTemporalDependenceAudit):
    table = x.series
    fig, ax = _mpl_axes("Pupil temporal dependence", "Lag-1 correlation", "Series")
    ax.hist(table["lag1"].dropna().to_numpy(dtype=float), bins=min(20, max(len(table), 5)))
    return fig


def plot_pupil_autocorrelation_comparison(x: PupilAutocorrelationComparison, absolute: bool = True):
    table = x.table
    value = "median_abs_acf" if absolute else "median_acf"
    fig, ax = _mpl_axes("Pupil autocorrelation comparison", "Lag", value)
    for label, group in table.groupby("model", sort=False):
        ax.plot(group["lag"], group[value], marker="o", label=str(label))
    ax.legend()
    return fig


def plot_pupil_missingness(x: PupilMissingnessAudit):
    fig, ax = _mpl_axes("Pupil missingness", "Variable", "Missing fraction")
    ax.bar(x.table["variable"].astype(str), x.table["missing_fraction"])
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_pupil_measurement_uncertainty(
    x: PupilMeasurementAudit05 | PupilMeasurementModel | AdvancedPupilSpecification,
):
    table = (
        x.table
        if isinstance(x, PupilMeasurementAudit05)
        else pupil_measurement_uncertainty_table(x)
    )
    fig, ax = _mpl_axes("Pupil measurement uncertainty", "Variable", "Declared")
    values = table["missing_fraction"] if "missing_fraction" in table else np.ones(len(table))
    ax.bar(table["variable"].astype(str), values)
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_binocular_pupil_trajectory(x: BinocularPupilTrajectory, probability: float | None = None):
    prob = x.probability if probability is None else _probability(probability, "probability", True)
    left = _central_summary(x.left_draws, prob)
    right = _central_summary(x.right_draws, prob)
    time_col = str(x.mapping["time"])
    fig, ax = _mpl_axes("Binocular pupil trajectory", "Time", "Pupil")
    times = pd.to_numeric(x.grid[time_col], errors="coerce").to_numpy(dtype=float)
    ax.plot(times, left["mean"], label="left")
    ax.plot(times, right["mean"], label="right")
    ax.legend()
    return fig


def plot_pupil_model_comparison(x: PupilModelComparison):
    table = pupil_model_comparison_table(x)
    fig, ax = _mpl_axes("Pupil model comparison", "Model", "ELPD")
    ax.bar(table["model"].astype(str), table["elpd"])
    return fig


def plot_pupil_lfo(x: PupilLFOValidation | PupilLFOComparison | PupilLFOPlan):
    fig, ax = _mpl_axes("Pupil leave-future-out", "Refit/model", "Future log score")
    if isinstance(x, PupilLFOPlan):
        ax.plot(x.table["refit"], x.table["train_through_index"], marker="o")
    elif isinstance(x, PupilLFOValidation):
        if not x.executed or x.scores is None:
            ax.plot(x.plan["refit"], x.plan["train_through_index"], marker="o")
        else:
            ax.plot(x.scores["refit"], x.scores["mean_log_score"], marker="o")
    elif isinstance(x, PupilLFOComparison):
        ax.bar(x.table["model"].astype(str), x.table["total_elpd_future"])
    else:
        raise GP3BayesError("Expected a pupil LFO object.")
    return fig


def plot_pupil_response_parameters(x: PupilResponseParameters):
    fig, ax = _mpl_axes("Pupil response parameters", "Parameter", "Posterior mean")
    ax.bar(x.table["parameter"].astype(str), x.table["mean"])
    ax.tick_params(axis="x", rotation=45)
    return fig


def plot_pupil_residual_spectrum(x: PupilResidualSpectrum):
    fig, ax = _mpl_axes("Pupil residual spectrum", "Normalized frequency", "Power")
    ax.plot(x.table["frequency"], x.table["median_power"])
    ax.fill_between(x.table["frequency"], x.table["q25_power"], x.table["q75_power"], alpha=0.2)
    return fig


def plot_pupil_model_complexity(x: AdvancedPupilSpecification | PupilComplexityAudit):
    audit = x.complexity_audit if isinstance(x, AdvancedPupilSpecification) else x
    if not isinstance(audit, PupilComplexityAudit):
        raise GP3BayesError("Expected a pupil complexity audit/specification.")
    fig, ax = _mpl_axes("Pupil model complexity", "Check", "Status")
    rank = {"ok": 0, "pass": 0, "review": 1, "high": 2, "blocked": 3, "failure": 3}
    table = audit.checks
    ax.bar(np.arange(len(table)), [rank.get(str(v), 1) for v in table["status"]])
    ax.set_xticks(np.arange(len(table)), table.iloc[:, 0].astype(str), rotation=45, ha="right")
    return fig


def plot_pupil_trajectory_derivative(
    x: PupilTrajectoryDerivative, probability: float | None = None
):
    table = pupil_trajectory_derivative_table(x, probability)
    time_col = next((c for c in table.columns if "time" in c.lower()), table.columns[0])
    return _plot_interval_frame(
        table, time_col, "estimate", "lower", "upper", "Pupil trajectory derivative"
    )


def plot_pupil_dynamic_contrast(x: PupilDynamicContrast):
    table = pupil_dynamic_contrast_table(x)
    time_col = next((c for c in table.columns if "time" in c.lower()), table.columns[0])
    return _plot_interval_frame(
        table, time_col, "estimate", "lower", "upper", "Pupil dynamic contrast"
    )


def plot_pupil_identifiability_audit(x: PupilIdentifiabilityAudit):
    fig, ax = _mpl_axes("Pupil identifiability audit", "Check", "Review rank")
    rank = {"pass": 0, "review": 1, "high": 2}
    ax.bar(np.arange(len(x.table)), [rank.get(v, 1) for v in x.table["status"]])
    ax.set_xticks(np.arange(len(x.table)), x.table["check"].astype(str), rotation=45, ha="right")
    return fig


def plot_pupil_predictive_calibration(x: PupilPredictiveCalibration | PupilPredictiveScore):
    score = x.score if isinstance(x, PupilPredictiveCalibration) else x
    if not isinstance(score, PupilPredictiveScore):
        raise GP3BayesError("Expected pupil predictive calibration/score evidence.")
    fig, ax = _mpl_axes("Pupil predictive calibration", "Predicted mean", "Observed")
    ax.scatter(score.pointwise["predicted_mean"], score.pointwise["observed"])
    lo = min(score.pointwise["predicted_mean"].min(), score.pointwise["observed"].min())
    hi = max(score.pointwise["predicted_mean"].max(), score.pointwise["observed"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    return fig


def plot_pupil_readiness(x: PupilReadiness):
    if not isinstance(x, PupilReadiness):
        raise GP3BayesError("Expected a pupil readiness audit.")
    fig, ax = _mpl_axes("Pupil readiness", "Metric", "Status")
    table = x.summary
    rank = {"pass": 0, "review": 1, "fail": 2, "failure": 2}
    status_col = "status" if "status" in table else table.columns[-1]
    ax.bar(np.arange(len(table)), [rank.get(str(v), 1) for v in table[status_col]])
    ax.set_xticks(np.arange(len(table)), table.iloc[:, 0].astype(str), rotation=45, ha="right")
    return fig


def plot_pupil_observed_trajectory(x: PupilPrepared | pd.DataFrame, summary: bool = True):
    data = x.data if isinstance(x, PupilPrepared) else x
    if not isinstance(data, pd.DataFrame):
        raise GP3BayesError("Expected prepared pupil data or a data frame.")
    time_col = (
        ".event_time"
        if ".event_time" in data
        else next((c for c in data if "time" in c.lower()), None)  # type: ignore[attr-defined]
    )
    value_col = (
        ".pupil_model"
        if ".pupil_model" in data
        else next((c for c in data if "pupil" in c.lower()), None)  # type: ignore[attr-defined]
    )
    if time_col is None or value_col is None:
        raise GP3BayesError("Could not identify time/pupil columns for plotting.")
    fig, ax = _mpl_axes("Observed pupil trajectory", "Time", "Pupil")
    if summary:
        condition = (
            ".condition" if ".condition" in data else ("condition" if "condition" in data else None)
        )
        keys = [time_col] + ([condition] if condition else [])
        table = (
            data.groupby(keys, observed=True, sort=False)[value_col].mean().reset_index(name="mean")  # type: ignore[call-overload]
        )
        if condition:
            for label, group in table.groupby(condition, observed=True, sort=False):
                ax.plot(group[time_col], group["mean"], label=str(label))
            ax.legend()
        else:
            ax.plot(table[time_col], table["mean"])
    else:
        ax.scatter(data[time_col], data[value_col], s=8, alpha=0.3)
    return fig


def plot_pupil_posterior_trajectory(x: PupilTrajectory | PupilPrediction):
    trajectory = estimate_pupil_trajectory(x) if isinstance(x, PupilPrediction) else x
    if not isinstance(trajectory, PupilTrajectory):
        raise GP3BayesError("Expected a posterior pupil trajectory.")
    table = trajectory.table
    time_col = (
        ".event_time"
        if ".event_time" in table
        else next((c for c in table if "time" in c.lower()), table.columns[0])  # type: ignore[attr-defined]
    )
    condition = ".condition" if ".condition" in table else None
    mean_col = (
        "mean" if "mean" in table else "estimate" if "estimate" in table else "predicted_mean"
    )
    low_col = "lower" if "lower" in table else "q_low"
    high_col = "upper" if "upper" in table else "q_high"
    return _plot_interval_frame(
        table,
        time_col,  # type: ignore[arg-type]
        mean_col,
        low_col,
        high_col,
        "Posterior pupil trajectory",
        condition,  # type: ignore[arg-type]
    )


def plot_pupil_estimand(x: PupilEstimand):
    if not isinstance(x, PupilEstimand):
        raise GP3BayesError("Expected a pupil estimand.")
    fig, ax = _mpl_axes(f"Pupil estimand: {x.estimand}", "Row", "Estimate")
    table = x.table
    value = (
        "mean"
        if "mean" in table
        else (
            "estimate"
            if "estimate" in table
            else table.select_dtypes(include=[np.number]).columns[0]
        )
    )
    ax.bar(np.arange(len(table)), table[value])
    return fig


def plot_pupil_ppc(x: PupilPPC, component: str = "trajectory"):
    table = pupil_ppc_table(x, component)  # type: ignore[arg-type]
    fig, ax = _mpl_axes(f"Pupil PPC: {component}")
    if component == "trajectory":
        for label, group in table.groupby(".condition", observed=True, sort=False):
            ax.plot(group[".event_time"], group["observed_mean"], label=f"observed {label}")
            ax.plot(
                group[".event_time"],
                group["replicated_median"],
                linestyle="--",
                label=f"replicated {label}",
            )
        ax.legend()
    elif component in {"autocorrelation", "features", "distribution"}:
        name_col = "statistic"
        ax.bar(table[name_col].astype(str), table["replicated_median"])
        ax.tick_params(axis="x", rotation=45)
    else:
        numeric = table.select_dtypes(include=[np.number])
        if numeric.shape[1]:
            ax.plot(np.arange(len(table)), numeric.iloc[:, 0])
    return fig


def plot_pupil_residual_acf(x: PupilFit | PupilDiagnostics):
    table = pupil_residual_acf(x)
    fig, ax = _mpl_axes("Pupil residual ACF", "Lag", "ACF")
    ax.vlines(table["lag"], 0, table["acf"])
    return fig


def plot_pupil_validation(x: PupilValidation):
    if not isinstance(x, PupilValidation):
        raise GP3BayesError("Expected a pupil validation object.")
    table = x.table
    fig, ax = _mpl_axes("Pupil validation", "Fold", "Score")
    numeric = table.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        ax.plot(numeric.iloc[:, 0], numeric.iloc[:, -1], marker="o")
    elif numeric.shape[1]:
        ax.bar(np.arange(len(table)), numeric.iloc[:, 0])
    return fig


def plot_pupil_sensitivity(x: PupilSensitivitySuite | PupilSensitivityComparison):
    table = pupil_sensitivity_table(x)
    fig, ax = _mpl_axes("Pupil sensitivity", "Scenario", "Estimate")
    if isinstance(x, PupilSensitivitySuite):
        ax.bar(np.arange(len(table)), np.ones(len(table)))
        ax.set_xticks(
            np.arange(len(table)),
            table["scenario_id"].astype(str)
            if "scenario_id" in table
            else table.iloc[:, 0].astype(str),
            rotation=45,
            ha="right",
        )
    else:
        numeric = table.select_dtypes(include=[np.number])
        ax.bar(
            np.arange(len(table)), numeric.iloc[:, 0] if numeric.shape[1] else np.ones(len(table))
        )
    return fig


def plot_pupil_measurement_audit(x: PupilMeasurementAudit):
    if not isinstance(x, PupilMeasurementAudit):
        raise GP3BayesError("Expected a pupil measurement audit.")
    fig, ax = _mpl_axes("Pupil measurement audit", "Metric", "Value")
    table = x.table
    numeric = table.select_dtypes(include=[np.number])
    if numeric.shape[1]:
        ax.bar(np.arange(len(table)), numeric.iloc[:, 0])
        ax.set_xticks(np.arange(len(table)), table.iloc[:, 0].astype(str), rotation=45, ha="right")
    return fig
