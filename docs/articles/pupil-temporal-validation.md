# Validation for temporally dependent pupil data

> Python-facing port of `pupil-temporal-validation.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Start with the prediction target

Temporally dependent samples should not automatically be treated as
exchangeable observation-level units. `create_pupil_validation_plan()` makes
the intended target explicit before computing validation.

Other supported targets distinguish a new participant and a future time
segment.

## Execution

Grouped K-fold uses complete validation groups. The future-segment target uses
a chronological holdout and explicit refitting rather than presenting
ordinary observation-wise PSIS-LOO as a universal time-series solution.
Validation answers only the declared predictive question.

## Python API mapping

- `gp3bayespy.create_pupil_contract`
- `gp3bayespy.create_pupil_validation_plan`
- `gp3bayespy.plot_pupil_validation`
- `gp3bayespy.prepare_pupil_timecourse`
- `gp3bayespy.pupil_validation_table`
- `gp3bayespy.simulate_pupil_timecourse`
- `gp3bayespy.validate_pupil_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
