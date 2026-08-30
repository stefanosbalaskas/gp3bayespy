from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gp3bayespy.exceptions import GP3BayesError

pe = importlib.import_module("gp3bayespy.postfit_exploration")


def test_probability_table_without_rope_branch():
    table = pe.posterior_probability_table(pd.DataFrame({"a": [-1.0, 0.0, 1.0]}))
    assert "probability_in_rope" not in table.columns


def test_group_effect_item_absent_and_flat_z_continue(monkeypatch):
    fit = SimpleNamespace(
        specification=SimpleNamespace(
            prepared=SimpleNamespace(data=pd.DataFrame({"pid": ["p1", "p2"]})),
            contract=SimpleNamespace(mappings={"participant": "pid", "item": None}),
        )
    )
    monkeypatch.setattr(
        pe,
        "_posterior_components",
        lambda x: {
            "sd_participant": np.ones((2, 3)),
            "participant_z": np.ones((2, 3)),
        },
    )
    with pytest.raises(GP3BayesError, match="No group-level effects"):
        pe.group_effect_table(fit)
