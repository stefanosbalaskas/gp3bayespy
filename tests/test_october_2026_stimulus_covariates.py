import numpy as np
import pandas as pd
import pytest

from research.october_2026.stimulus_covariates import (
    extract_gaze_contingent_features,
    pupil_adjustment_design_audit,
)


def test_attended_brightness_differs_from_global():
    frame = np.zeros((7, 7), dtype=float)
    frame[:, 3:] = 10
    result = extract_gaze_contingent_features(
        np.stack([frame, frame]), [[0., 0.], [1., 1.]], radius_px=1,
    )
    assert result.iloc[0].attended_patch_brightness == 0
    assert result.iloc[1].attended_patch_brightness == 10
    assert result.iloc[0].global_brightness == pytest.approx(
        result.iloc[1].global_brightness)


def test_rejects_bad_geometry():
    with pytest.raises(ValueError, match="normalized gaze"):
        extract_gaze_contingent_features(np.ones((2, 4, 4)), [[-0.1, 0], [0.5, 0.5]])


def test_model_adjustment_descriptive_contrast_and_collinearity():
    rows = []
    for participant in ["a", "b", "c"]:
        for k in range(10):
            cond = k % 2
            bright = (k % 3) / 4
            complexity = (k % 4) / 5
            pupil = 1 + .4 * cond + .3 * bright - .2 * complexity
            rows.append([participant, cond, pupil, bright, complexity])
    d = pd.DataFrame(rows, columns=[
        "participant_id", "condition", "pupil",
        "attended_patch_brightness", "attended_orientation_entropy_bits"])
    result = pupil_adjustment_design_audit(d)
    assert result["condition_contrast_adjusted"] == pytest.approx(.4)
    d["attended_orientation_entropy_bits"] = d["attended_patch_brightness"]
    with pytest.raises(ValueError, match="collinear"):
        pupil_adjustment_design_audit(d)
