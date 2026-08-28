# Synthetic Gazepoint pupillometry case study

> Python-facing port of `synthetic-gazepoint-pupillometry-case-study.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Purpose

This case study is **entirely synthetic**. Its statistics are software
demonstrations and are not empirical evidence about Gazepoint hardware,
participants, or psychological processes.

## A Gazepoint-like mapping audit

The raw simulator is vendor-neutral. The next object creates a small
Gazepoint-like view solely to exercise the verified field bridge.

`LPD` is labelled as pixels by the bridge. The example does not convert those
synthetic values into millimetres.

## Contract through specification

## Backend and post-fit stages

## Sensitivity

All windows and sensitivity dimensions are declared. No simulated estimate is
presented as an empirical effect, and no scenario is automatically selected.

## Python API mapping

- `gp3bayespy.audit_pupil_measurement_context`
- `gp3bayespy.audit_pupil_readiness`
- `gp3bayespy.check_pupil_posterior_predictive`
- `gp3bayespy.create_pupil_contract`
- `gp3bayespy.create_pupil_sensitivity_suite`
- `gp3bayespy.create_pupil_validation_plan`
- `gp3bayespy.diagnose_pupil_fit`
- `gp3bayespy.estimate_pupil_auc`
- `gp3bayespy.estimate_pupil_trajectory`
- `gp3bayespy.estimate_pupil_window`
- `gp3bayespy.fit_pupil_model_backend`
- `gp3bayespy.gazepoint_pupil_mapping_table`
- `gp3bayespy.inspect_gazepoint_pupil_schema`
- `gp3bayespy.predict_pupil_trajectory`
- `gp3bayespy.prepare_pupil_timecourse`
- `gp3bayespy.pupil_measurement_audit_table`
- `gp3bayespy.pupil_readiness_table`
- `gp3bayespy.pupil_sensitivity_table`
- `gp3bayespy.pupil_specification_table`
- `gp3bayespy.simulate_pupil_timecourse`
- `gp3bayespy.specify_pupil_timecourse_model`
- `gp3bayespy.validate_pupil_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
