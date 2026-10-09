import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

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
