"""Experimental gaze-contingent visual covariates for pupil-model sensitivity.

No claim that orientation entropy is a validated universal pupil correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def extract_gaze_contingent_features(
    grayscale_frames,
    gaze_xy,
    *,
    radius_px: int = 6,
    orientation_bins: int = 12,
) -> pd.DataFrame:
    """Extract local brightness and orientation entropy at aligned gaze samples.

    Frames have shape [time, height, width] and must be linear/declared
    stimulus brightness, not unqualified display luminance. Gaze coordinates
    are normalized to [0,1]. This does not estimate retinal luminance, display
    gamma, viewing angle, or light reaching the eye.
    """
    frames = np.asarray(grayscale_frames, dtype=float)
    gaze = np.asarray(gaze_xy, dtype=float)
    if frames.ndim != 3 or gaze.shape != (len(frames), 2):
        raise ValueError("frames [time,height,width] and gaze [time,2] required")
    if not np.isfinite(frames).all() or not np.isfinite(gaze).all():
        raise ValueError("frames and gaze coordinates must be finite")
    if np.any((gaze < 0) | (gaze > 1)):
        raise ValueError("normalized gaze must remain in [0,1]")
    if frames.shape[1] < 3 or frames.shape[2] < 3:
        raise ValueError("image height and width must be at least three")
    if radius_px < 1 or orientation_bins < 3:
        raise ValueError("radius_px>=1 and at least three orientation bins required")
    rows = []
    h, w = frames.shape[1:]
    for i, (frame, xy) in enumerate(zip(frames, gaze)):
        x = int(round(xy[0] * (w - 1)))
        y = int(round(xy[1] * (h - 1)))
        patch = frame[max(0, y - radius_px):min(h, y + radius_px + 1),
                      max(0, x - radius_px):min(w, x + radius_px + 1)]
        global_mean = float(frame.mean())
        local_mean = float(patch.mean())
        orientation_entropy = np.nan
        if min(patch.shape) >= 2:
            gy, gx = np.gradient(patch)
            strength = np.hypot(gx, gy)
            active = strength > 1e-12
            if active.any():
                angle = np.mod(np.arctan2(gy[active], gx[active]), np.pi)
                bins = np.histogram(angle, bins=orientation_bins,
                                    range=(0, np.pi))[0].astype(float)
                probs = bins[bins > 0] / bins.sum()
                orientation_entropy = float(-(probs * np.log2(probs)).sum())
        rows.append({"frame_index": i, "global_brightness": global_mean,
                     "attended_patch_brightness": local_mean,
                     "attended_orientation_entropy_bits": orientation_entropy,
                     "patch_n_pixels": int(patch.size)})
    return pd.DataFrame(rows)


def pupil_adjustment_design_audit(
    frame: pd.DataFrame,
    *,
    participant_col: str = "participant_id",
    condition_col: str = "condition",
    outcome_col: str = "pupil",
    covariate_cols=("attended_patch_brightness",
                    "attended_orientation_entropy_bits"),
) -> dict:
    """Describe raw and adjusted condition contrasts with participant intercepts.

    Ordinary least squares sensitivity ONLY; no Bayesian posterior, causal
    estimand, or uncertainty claim. Refuses rank-deficient designs.
    """
    required = [participant_col, condition_col, outcome_col, *covariate_cols]
    if not set(required).issubset(frame.columns):
        raise ValueError("missing design columns")
    if frame[participant_col].isna().any():
        raise ValueError("participant identities must be observed")
    d = frame[required].copy()
    numeric = d[[condition_col, outcome_col, *covariate_cols]].apply(
        pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("missing/undefined pupil or stimulus predictor")
    cond = numeric[condition_col].to_numpy(dtype=float)
    if not np.isin(cond, [0., 1.]).all() or len(np.unique(cond)) != 2:
        raise ValueError("condition must be observed binary zero/one")
    ids = pd.get_dummies(d[participant_col].astype(str), drop_first=True).to_numpy(dtype=float)
    intercept = np.ones((len(d), 1))
    base = np.column_stack((intercept, ids, cond))
    covariates = numeric[list(covariate_cols)].to_numpy(dtype=float)
    adjusted = np.column_stack((base, covariates))
    if np.linalg.matrix_rank(base) < base.shape[1] or (
            np.linalg.matrix_rank(adjusted) < adjusted.shape[1]):
        raise ValueError("collinear design; stimulus adjustment not identifiable")
    outcome = numeric[outcome_col].to_numpy(dtype=float)
    beta_base = np.linalg.lstsq(base, outcome, rcond=None)[0]
    beta_adj = np.linalg.lstsq(adjusted, outcome, rcond=None)[0]
    return {
        "condition_contrast_unadjusted": float(beta_base[-1]),
        "condition_contrast_adjusted": float(beta_adj[base.shape[1] - 1]),
        "change_after_adjustment": float(beta_adj[base.shape[1] - 1] - beta_base[-1]),
        "n_rows": len(d), "n_participants": d[participant_col].nunique(),
        "claim_boundary": "Descriptive OLS design sensitivity only; not a Bayesian/causal inference.",
    }
