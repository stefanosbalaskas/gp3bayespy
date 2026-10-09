import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.methods_briefing_2026.nonlinear_mediation_audit import (  # noqa: E402
    audit_mediation_functional_form,
)


def _sample():
    rng = np.random.default_rng(2026)
    participants = np.repeat(np.arange(24), 6)
    x = np.tile([0, 1, 0, 1, 0, 1], 24).astype(float)
    m = .4 + .8 * x + rng.normal(0, .4, len(x))
    y = 1 + .3 * x + .7 * m + .3 * m ** 2 + rng.normal(0, .3, len(x))
    return pd.DataFrame({"participant_id": participants, "x": x, "m": m, "y": y})


def test_comparison_and_cluster_bootstrap_are_seeded():
    kw = dict(treatment_col="x", mediator_col="m", outcome_col="y", bootstrap_replicates=20)
    first = audit_mediation_functional_form(_sample(), **kw)
    second = audit_mediation_functional_form(_sample(), **kw)
    assert first["n_participants"] == 24
    assert first["estimates"]["model"].tolist() == ["linear", "quadratic"]
    pd.testing.assert_frame_equal(first["bootstrap_ledger"], second["bootstrap_ledger"])
    assert first["bootstrap_ledger"].shape[0] == 20
    assert "Not a Bayesian posterior" in first["claim_boundary"]


def test_nonbinary_treatment_and_small_cluster_count_are_rejected():
    d = _sample()
    d.loc[0, "x"] = 2
    with pytest.raises(ValueError, match="0/1"):
        audit_mediation_functional_form(
            d, treatment_col="x", mediator_col="m", outcome_col="y"
        )
    d = _sample().iloc[:12]
    with pytest.raises(ValueError, match="four independent"):
        audit_mediation_functional_form(
            d, treatment_col="x", mediator_col="m", outcome_col="y"
        )


def test_known_truth_quadratic_mediator_shift_is_recovered():
    """Balanced, noiseless known-truth benchmark with an analytic contrast."""
    x = np.tile([0.0] * 5 + [1.0] * 5, 40)
    residual = np.tile([-0.4, -0.2, 0.0, 0.2, 0.4] * 2, 40)
    m = 0.4 + 0.7 * x + residual
    y = 1.0 + 0.2 * x + 0.6 * m + 0.4 * m**2
    data = pd.DataFrame({
        "participant_id": np.repeat(np.arange(40), 10),
        "x": x, "m": m, "y": y,
    })
    res = audit_mediation_functional_form(
        data,
        treatment_col="x", mediator_col="m", outcome_col="y",
        bootstrap_replicates=30, seed=17,
    )
    fitted = res["estimates"].set_index("model")["model_based_indirect_contrast"]
    # E[(.6 * (m0+.7) + .4 * (m0+.7)^2) - (.6*m0 + .4*m0^2)]
    # = .6*.7 + .4*(2*.4*.7 + .7**2) = .84.
    assert fitted["quadratic"] == pytest.approx(0.84, abs=1e-10)
    assert abs(fitted["linear"] - fitted["quadratic"]) > 0.1
    assert (res["estimates"]["bootstrap_successes"] == 30).all()


def test_known_truth_null_mediator_shift_is_zero():
    x = np.tile([0.0] * 5 + [1.0] * 5, 40)
    residual = np.tile([-0.4, -0.2, 0.0, 0.2, 0.4] * 2, 40)
    m = 0.4 + residual  # treatment does not move mediator
    y = 1.0 + 0.2 * x + 0.6 * m + 0.4 * m**2
    d = pd.DataFrame({
        "participant_id": np.repeat(np.arange(40), 10),
        "x": x, "m": m, "y": y,
    })
    r = audit_mediation_functional_form(
        d, treatment_col="x", mediator_col="m", outcome_col="y",
        bootstrap_replicates=0,
    )
    assert np.allclose(r["estimates"]["model_based_indirect_contrast"], 0, atol=1e-10)
    assert (r["estimates"]["bootstrap_successes"] == 0).all()
    assert (r["estimates"]["bootstrap_failures"] == 0).all()


def test_unmeasured_mediator_outcome_common_cause_breaks_causal_interpretation():
    """Known-truth stress: precise model-based fit can be causally biased."""
    x = np.tile([0.0] * 5 + [1.0] * 5, 40)
    common_cause = np.tile([-0.4, -0.2, 0.0, 0.2, 0.4] * 2, 40)
    m = 0.4 + 0.7 * x + common_cause
    y = 1.0 + 0.2 * x + 0.6 * m + 0.4 * m**2 + 0.5 * common_cause
    d = pd.DataFrame({
        "participant_id": np.repeat(np.arange(40), 10),
        "x": x, "m": m, "y": y,
    })
    result = audit_mediation_functional_form(
        d, treatment_col="x", mediator_col="m", outcome_col="y",
        bootstrap_replicates=0,
    )
    # True intervention on M, holding U fixed: .84. But conditioning on
    # observational M transmits .5 of unobserved U per unit of M.
    implied = result["estimates"].set_index("model").loc[
        "quadratic", "model_based_indirect_contrast"
    ]
    assert implied == pytest.approx(1.19, abs=1e-10)
    assert abs(implied - 0.84) == pytest.approx(0.35, abs=1e-10)
    assert "not a Bayesian posterior" in result["claim_boundary"].lower()
