from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from gp3bayespy.exceptions import GP3BayesError

aow = importlib.import_module("gp3bayespy.advanced_optional_workflows")
hea = importlib.import_module("gp3bayespy.hierarchical_effects_advanced")


def test_advanced_optional_actual_guard_paths():
    base = SimpleNamespace(marker="ok")
    wrapped = aow.InteractionPriorSpecification(
        base=base,
        advanced_priors={
            "main_effect_scale": 0.5,
            "interaction_scale": 0.25,
            "interaction": ("x", "z"),
        },
    )
    assert wrapped.marker == "ok"

    translated = aow.InteractionBackendSpecification(base=base, interaction_scale=0.25)
    assert translated.marker == "ok"

    no_interaction = SimpleNamespace(contract=SimpleNamespace(interaction=None))
    with pytest.raises(GP3BayesError):
        aow.specify_binary_model_with_interaction_prior(no_interaction, baseline=0.5)
    with pytest.raises(GP3BayesError):
        aow.specify_duration_model_with_interaction_prior(no_interaction, baseline=500.0)

    interaction = SimpleNamespace(contract=SimpleNamespace(interaction=("x", "z")))
    with pytest.raises(GP3BayesError):
        aow.specify_binary_model_with_interaction_prior(
            interaction, baseline=0.5, main_effect_scale=0.0
        )
    with pytest.raises(GP3BayesError):
        aow.specify_duration_model_with_interaction_prior(
            interaction, baseline=500.0, interaction_scale=0.0
        )

    with pytest.raises(GP3BayesError):
        aow.interaction_prior_summary(object())


def test_hierarchical_effects_actual_integer_and_level_paths():
    with pytest.raises(GP3BayesError):
        hea._integer(True, "x")
    with pytest.raises(GP3BayesError):
        hea._integer(0, "x")
    assert hea._integer(2, "x") == 2

    bare = SimpleNamespace(specification=None)
    assert hea._levels(bare, "participant", 2) == ["1", "2"]

    with pytest.raises(GP3BayesError):
        hea._group_arrays(SimpleNamespace())
