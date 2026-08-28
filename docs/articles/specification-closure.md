# Specification Closure: Strict Readiness and Governed Validation

> Python-facing port of `specification-closure.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Purpose

This article closes the remaining Phase-0 validation requirements without
expanding gp3bayes beyond its two approved families. The new checks are
observable-data diagnostics. They do not establish posterior adequacy, choose
a model automatically, or justify deleting observations.

## Binary strict readiness

The strict audit adds explicit overall condition imbalance, participant outcome
variation, identifier-like predictor review, and fixed-effect rank checks. When
`detectseparation` is installed, the optional separation screen can also be
integrated by setting `run_separation = TRUE`.

## Identifier-like predictors are review signals

The heuristic never silently removes a declared predictor. A flag means that
the analyst must verify whether the numeric column is substantively meaningful
or is an identifier accidentally entered into the model matrix.

## Duration extremes, impossible ranges, and censoring

Extreme values remain in the data. Censoring and impossible-range violations
are contract failures for the positive uncensored lognormal workflow; they do
not trigger an automatic switch to another likelihood.

## Traceability

The table is intended to make specification closure auditable: every remaining
Phase-0 requirement has an explicit implementation point and all automatic
decision flags remain `FALSE`.

## Python API mapping

- `gp3bayespy.audit_duration_boundaries`
- `gp3bayespy.audit_model_readiness_strict`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.gp3bayes_specification_traceability`
- `gp3bayespy.identify_identifier_like_predictors`
- `gp3bayespy.review_duration_extremes`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.simulate_hierarchical_duration_data`
- `gp3bayespy.summarise_binary_group_variation`
- `gp3bayespy.summarise_condition_balance`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
