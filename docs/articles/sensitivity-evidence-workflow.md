# Unified Sensitivity Suites and Evidence Inventories

> Python-facing port of `sensitivity-evidence-workflow.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Orchestration without automatic robustness claims

`gp3bayespy` already provides prior sensitivity, power scaling, PSIS-LOO,
structural sensitivity, group-deletion sensitivity, coding/scaling variants,
duration-unit invariance and exact K-fold validation. Version 0.2.0 adds a
thin orchestration layer so these results can be planned and collected without
turning them into an automatic "robust/not robust" verdict.

## Declare a suite before running it

Creating the plan runs **nothing**. Expensive components only run when
`run_sensitivity_suite()` receives both a fitted model and an explicit plan.

Structural sensitivity can be declared using the package's existing governed
plans:

## Evidence is an inventory

Already-computed results can be collected into one review object.

Reports require an explicit file path:

The inventory deliberately withholds aggregate adequacy, robustness, causal,
and model-selection claims. Different evidence components answer different
questions and can disagree without being collapsed into a single score.

## Python API mapping

- `gp3bayespy.collect_model_evidence`
- `gp3bayespy.create_group_deletion_sensitivity_plan`
- `gp3bayespy.create_model_evidence_report`
- `gp3bayespy.create_random_slope_sensitivity_plan`
- `gp3bayespy.create_sensitivity_suite_plan`
- `gp3bayespy.run_sensitivity_suite`
- `gp3bayespy.summarise_sensitivity_suite`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/sensitivity_workflow.py`.
