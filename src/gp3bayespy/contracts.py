"""Contract-first model declarations.

This module is a Python-native port of the frozen gp3bayes 0.5.0
``model-contract.R`` contract layer.  Creating a contract does not validate
input data and never implies model adequacy, convergence, or causal validity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .exceptions import GP3BayesError

_MODEL_FAMILIES = {
    "binary": "hierarchical_binary",
    "duration": "hierarchical_lognormal_duration",
}

_BINARY_TEMPLATE: dict[str, Any] = {
    "intended_outcome": "Binary trial-level event",
    "outcome_scale": "0 or 1",
    "unit_of_analysis": "One participant-trial or participant-item row",
    "grouping_structure": (
        "Participant random intercept required",
        "Optional crossed item random intercept",
        "At most one participant random slope for the focal condition",
    ),
    "repeated_measures_structure": (
        "Repeated trial-level observations within participant",
        "Optional crossed item or stimulus observations",
        "No automatic serial-correlation model",
    ),
    "supported_predictors": (
        "Focal condition",
        "Optional linear time or trial-order term",
        "Prespecified participant-, item-, or trial-level predictors",
    ),
    "supported_interactions": (
        "At most one prespecified two-way interaction among declared predictors"
    ),
    "supported_offsets_or_exposures": "Not supported",
    "supported_censoring": "Not applicable",
    "likelihood": "Bernoulli",
    "link": "logit",
    "estimands": (
        "Population-level conditional log-odds contrast",
        "Conditional odds ratio",
        "Design-standardised predicted-probability contrast",
    ),
    "coefficient_interpretation": (
        "Population-level conditional log-odds contrast per declared predictor unit"
    ),
    "prior_families": (
        "Normal prior for the intercept on the logit scale",
        "Normal priors for population-level coefficients",
        "Half-Student-t priors for group-level standard deviations",
        "LKJ prior for a supported random-effect correlation",
    ),
    "prior_rationale": (
        "The intercept prior must encode plausible baseline event probabilities on the logit scale",
        (
            "Coefficient priors regularise implausibly large log-odds contrasts while "
            "retaining substantive uncertainty"
        ),
        "Group-level scale priors constrain extreme heterogeneity without fixing it to zero",
        "The correlation prior regularises an included intercept-slope correlation",
    ),
    "scaling_expectations": (
        "No silent scaling",
        "Record all centring and scaling decisions",
        "Identifiers must not be used as numeric predictors",
    ),
    "assumptions": (
        "Conditional Bernoulli sampling is appropriate",
        "The approved grouping structure represents repeated observations",
        "The design supports the requested contrast",
        "Important serial dependence is not left unmodelled",
    ),
    "convergence_criteria": (
        "R-hat no greater than 1.01",
        "Adequate bulk and tail effective sample sizes",
        "Zero divergent transitions",
        "Zero maximum-treedepth saturations",
        "Acceptable energy and chain-mixing diagnostics",
    ),
    "prior_predictive_checks": (
        "Overall and condition-specific event rates",
        "Participant-level and item-level rate dispersion",
        "Implausibly extreme probability or contrast frequency",
    ),
    "posterior_predictive_checks": (
        "Overall and condition-specific event rates",
        "Participant-level and item-level rate distributions",
        "Sparse-cell and all-zero or all-one pattern frequencies",
    ),
    "sensitivity_requirements": (
        "Population-level coefficient prior scales",
        "Group-level standard-deviation priors",
        "Random-intercept versus approved random-slope specification",
        "Influential participant and item cases",
    ),
    "limitations": (
        "No automatic model selection",
        "No causal interpretation without an identifying design",
        "No transition, time-course, or arbitrary nonlinear structure",
    ),
    "unsupported_uses": (
        "Multinomial or ordinal outcomes",
        "Aggregated proportions without denominators",
        "Survey weights or automatic variable selection",
        "Psychological or protected-attribute inference",
    ),
    "interpretation_boundaries": (
        "Contract creation does not establish model adequacy or convergence",
        "Associations are not causal effects without an identifying design",
        (
            "Behavioural measurements do not directly identify latent psychological or "
            "protected attributes"
        ),
    ),
    "computational_requirements": (
        "Contract creation is backend-independent; fitting requires an approved optional "
        "backend"
    ),
}

_DURATION_TEMPLATE: dict[str, Any] = {
    "intended_outcome": "Strictly positive uncensored duration",
    "outcome_scale": "Finite continuous value greater than zero",
    "unit_of_analysis": "One participant-trial, participant-item, or event row",
    "grouping_structure": _BINARY_TEMPLATE["grouping_structure"],
    "repeated_measures_structure": (
        "Repeated duration observations within participant",
        "Optional crossed item or stimulus observations",
        "No automatic serial-correlation model",
    ),
    "supported_predictors": _BINARY_TEMPLATE["supported_predictors"],
    "supported_interactions": _BINARY_TEMPLATE["supported_interactions"],
    "supported_offsets_or_exposures": "Not supported",
    "supported_censoring": "Not supported; durations must be strictly positive and uncensored",
    "likelihood": "lognormal",
    "link": "identity on mean log duration",
    "estimands": (
        "Population-level contrast in expected log duration",
        "Conditional median ratio",
        "Design-standardised predictive median difference or ratio",
        "Prespecified posterior predictive upper quantile",
    ),
    "coefficient_interpretation": (
        "Population-level contrast on the log-duration scale; exponentiation gives a "
        "conditional median ratio"
    ),
    "prior_families": (
        "Normal prior for log baseline median",
        "Normal priors for population-level coefficients",
        "Half-Student-t priors for residual and group-level scales",
        "LKJ prior for a supported random-effect correlation",
    ),
    "prior_rationale": (
        (
            "The intercept prior must reflect a plausible baseline median duration in the "
            "recorded unit after log transformation"
        ),
        "Coefficient priors regularise implausible multiplicative duration contrasts",
        (
            "Residual and group-level scale priors constrain implausible dispersion without "
            "fixing variability to zero"
        ),
        "The correlation prior regularises an included intercept-slope correlation",
    ),
    "scaling_expectations": (
        "Record the duration unit",
        "No silent unit conversion",
        "Record all centring and scaling decisions",
        "Identifiers must not be used as numeric predictors",
    ),
    "assumptions": (
        "Durations are strictly positive, finite, and uncensored",
        "Conditional log durations are adequately Gaussian",
        "The approved grouping structure represents repeated observations",
        "No important mixture, deadline, or serial process is omitted",
    ),
    "convergence_criteria": _BINARY_TEMPLATE["convergence_criteria"],
    "prior_predictive_checks": (
        "Median and interquartile range",
        "Upper quantiles and tail exceedance",
        "Implausibly small or large duration frequency",
        "Participant-level and item-level dispersion",
    ),
    "posterior_predictive_checks": (
        "Raw-scale and log-scale distribution",
        "Condition-specific medians and upper quantiles",
        "Participant-level and item-level distributions",
        "Within-participant condition contrasts",
    ),
    "sensitivity_requirements": (
        "Population-level coefficient prior scales",
        "Residual and group-level scale priors",
        "Random-intercept versus approved random-slope specification",
        "Influential participant and item cases",
        "Duration-unit rescaling invariance",
    ),
    "limitations": (
        "No automatic distribution switching",
        "No causal interpretation without an identifying design",
        "No time-course, autocorrelation, or distributional regression",
    ),
    "unsupported_uses": (
        "Zero, censored, or truncated durations",
        "Shifted-lognormal, Gamma, Weibull, survival, or mixture models",
        "Automatic variable selection",
        "Psychological or protected-attribute inference",
    ),
    "interpretation_boundaries": (
        "Contract creation does not establish model adequacy or convergence",
        (
            "Exponentiated coefficients are conditional median ratios and not automatically "
            "raw-mean ratios"
        ),
        "Associations are not causal effects without an identifying design",
        (
            "Behavioural measurements do not directly identify latent psychological or "
            "protected attributes"
        ),
    ),
    "computational_requirements": _BINARY_TEMPLATE["computational_requirements"],
}


def _match_contract_family(family: object) -> str:
    if not isinstance(family, str):
        raise GP3BayesError("`family` must be one non-missing character value.")
    if family not in _MODEL_FAMILIES:
        allowed = ", ".join(_MODEL_FAMILIES)
        raise GP3BayesError(
            f"Unsupported `family`: {family}. Supported values are: {allowed}."
        )
    return family


def _nonempty_name(
    value: object,
    argument: str,
    *,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value:
        raise GP3BayesError(
            f"`{argument}` must be one non-empty character value."
        )
    return value


def _unique_strings(value: Sequence[str], argument: str) -> tuple[str, ...]:
    values: tuple[str, ...]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(value)
    else:
        raise GP3BayesError(
            f"`{argument}` must be a character vector of unique, non-empty values."
        )

    if (
        any(not isinstance(item, str) or not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise GP3BayesError(
            f"`{argument}` must be a character vector of unique, non-empty values."
        )
    return values


@dataclass(frozen=True, slots=True)
class ModelContract:
    """Approved backend-independent model contract."""

    contract_version: str
    family: str
    model_family: str
    mappings: Mapping[str, str | None]
    predictors: tuple[str, ...]
    interaction: tuple[str, str] | None
    random_slope: bool
    outcome_unit: str | None
    notes: tuple[str, ...]
    template: Mapping[str, Any] = field(repr=False)

    def __getattr__(self, name: str) -> Any:
        template = object.__getattribute__(self, "template")
        if name in template:
            return template[name]
        raise AttributeError(name)

    @property
    def likelihood(self) -> str:
        return str(self.template["likelihood"])

    @property
    def link(self) -> str:
        return str(self.template["link"])

    @property
    def prior_rationale(self) -> tuple[str, ...]:
        return tuple(self.template["prior_rationale"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "family": self.family,
            "model_family": self.model_family,
            "mappings": dict(self.mappings),
            "predictors": list(self.predictors),
            "interaction": (
                None if self.interaction is None else list(self.interaction)
            ),
            "random_slope": self.random_slope,
            "outcome_unit": self.outcome_unit,
            "notes": list(self.notes),
            **dict(self.template),
        }

    def __repr__(self) -> str:
        lines = [
            "<gp3bayes_model_contract>",
            f"  Family: {self.family}",
            f"  Likelihood: {self.likelihood}",
            f"  Link: {self.link}",
            f"  Outcome: {self.mappings['outcome']}",
            f"  Participant: {self.mappings['participant']}",
        ]
        if self.mappings.get("item") is not None:
            lines.append(f"  Item: {self.mappings['item']}")
        if self.mappings.get("condition") is not None:
            lines.append(f"  Condition: {self.mappings['condition']}")
        if self.outcome_unit is not None:
            lines.append(f"  Outcome unit: {self.outcome_unit}")
        lines.extend(
            [
                f"  Random slope requested: {str(self.random_slope).upper()}",
                "  Fitting performed: FALSE",
            ]
        )
        return "\n".join(lines)


def create_model_contract(
    family: str,
    outcome_col: str,
    participant_col: str,
    item_col: str | None = None,
    trial_col: str | None = None,
    condition_col: str | None = None,
    time_col: str | None = None,
    predictors: Sequence[str] = (),
    interaction: Sequence[str] | None = None,
    random_slope: bool = False,
    outcome_unit: str | None = None,
    notes: Sequence[str] = (),
) -> ModelContract:
    """Create an approved backend-independent Bayesian model contract."""
    family_value = _match_contract_family(family)
    mappings: dict[str, str | None] = {
        "outcome": _nonempty_name(outcome_col, "outcome_col"),
        "participant": _nonempty_name(participant_col, "participant_col"),
        "item": _nonempty_name(item_col, "item_col", optional=True),
        "trial": _nonempty_name(trial_col, "trial_col", optional=True),
        "condition": _nonempty_name(
            condition_col, "condition_col", optional=True
        ),
        "time": _nonempty_name(time_col, "time_col", optional=True),
    }

    predictor_values = _unique_strings(predictors, "predictors")
    note_values = _unique_strings(notes, "notes")

    interaction_values: tuple[str, str] | None = None
    if interaction is not None:
        values = _unique_strings(interaction, "interaction")
        if len(values) != 2:
            raise GP3BayesError(
                "`interaction` must contain exactly two declared variables."
            )
        available = {
            value
            for value in (
                mappings["condition"],
                mappings["time"],
                *predictor_values,
            )
            if value is not None
        }
        if not set(values).issubset(available):
            raise GP3BayesError(
                "Every interaction variable must be declared through "
                "`condition_col`, `time_col`, or `predictors`."
            )
        interaction_values = (values[0], values[1])

    if not isinstance(random_slope, bool):
        raise GP3BayesError("`random_slope` must be TRUE or FALSE.")
    if random_slope and mappings["condition"] is None:
        raise GP3BayesError(
            "`condition_col` must be supplied when `random_slope = TRUE`."
        )

    declared = [
        value for value in mappings.values() if value is not None
    ] + list(predictor_values)
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in declared:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise GP3BayesError(
            "Column mappings and predictors must be unique. Duplicated: "
            + ", ".join(duplicates)
            + "."
        )

    if family_value == "binary":
        if outcome_unit is not None:
            raise GP3BayesError(
                "`outcome_unit` must be NULL for the binary family."
            )
        outcome_unit_value = None
        template = _BINARY_TEMPLATE
    else:
        outcome_unit_value = _nonempty_name(outcome_unit, "outcome_unit")
        template = _DURATION_TEMPLATE

    return ModelContract(
        contract_version="0.1",
        family=family_value,
        model_family=_MODEL_FAMILIES[family_value],
        mappings=mappings,
        predictors=predictor_values,
        interaction=interaction_values,
        random_slope=random_slope,
        outcome_unit=outcome_unit_value,
        notes=note_values,
        template=template,
    )
