# Governed Predictive Model Comparison

> Python-facing port of `governed-pupil-model-comparison.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The sensitivity suite is a pre-fit registry of alternatives. It does not fit or rank models.

Weights are returned only as explicit evidence. gp3bayes does not automatically average predictions or declare the highest-weight model substantively correct.

# Leave-future-out is an explicit refit workflow

## Python API mapping

- `gp3bayespy.compare_pupil_models`
- `gp3bayespy.create_pupil_advanced_sensitivity_suite`
- `gp3bayespy.create_pupil_lfo_plan`
- `gp3bayespy.create_pupil_model_set`
- `gp3bayespy.fit_advanced_pupil_model_backend`
- `gp3bayespy.materialize_pupil_advanced_sensitivity_scenario`
- `gp3bayespy.plot_pupil_lfo`
- `gp3bayespy.plot_pupil_model_comparison`
- `gp3bayespy.pupil_model_comparison_table`
- `gp3bayespy.pupil_model_weights`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`
- `gp3bayespy.validate_pupil_leave_future_out`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
