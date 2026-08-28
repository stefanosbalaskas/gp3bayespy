# Preparing and auditing pupil time courses

> Python-facing port of `pupil-preparation-and-auditing.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Preparation is not preprocessing automation

`prepare_pupil_timecourse()` validates declared columns and records only
transformations explicitly requested by the analyst. It does not detect or
repair blinks, interpolate missing values, smooth traces, select an eye, or
choose a baseline window.

## Explicit baseline transformation

A baseline operation is performed only when requested, and the declared
baseline window must be available. Data already declared as baseline-adjusted
cannot be baseline-adjusted a second time.

The raw declared pupil value remains linked to the model value in the
prepared object.

## Readiness evidence

The audit reports sample support, sampling intervals, missingness, baseline
coverage, indicators, gaze/luminance availability, and related measurement
context. It does not remove observations.

## Measurement-context audit

PFE status is carried from the contract. Gaze coordinates are evidence about
measurement context and can be declared as nuisance covariates or used in
sensitivity scenarios, but this foundation does not implement a universal PFE
correction.

## Python API mapping

- `gp3bayespy.audit_pupil_measurement_context`
- `gp3bayespy.audit_pupil_readiness`
- `gp3bayespy.create_pupil_contract`
- `gp3bayespy.prepare_pupil_timecourse`
- `gp3bayespy.pupil_measurement_audit_table`
- `gp3bayespy.pupil_readiness_table`
- `gp3bayespy.simulate_pupil_timecourse`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
