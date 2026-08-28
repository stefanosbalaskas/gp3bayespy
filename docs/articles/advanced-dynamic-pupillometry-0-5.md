# Advanced Dynamic Pupillometry in gp3bayes 0.5

> Python-facing port of `advanced-dynamic-pupillometry-0-5.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Scope

gp3bayespy (R reference 0.5) extends the governed 0.4 dynamic-pupillometry foundation without replacing its data contract. The advanced layer makes the observation distribution, residual scale, temporal dependence, temporal function class, measurement uncertainty, missingness assumptions, and predictive target explicit components of the model specification.

The layer remains deliberately narrow. It does not interpolate blinks, infer cognitive states, identify causal effects, declare missingness mechanisms true, or choose a preferred model automatically.

# A backend-free advanced workflow

The simulator stores the generating truth separately from the observed data. It is intended for examples and validation rather than as a physiological pupil generator.

# Declare an advanced model

No Stan model has been compiled or fitted at this point.

# Translate, fit, diagnose, and estimate

Translation can be inspected without fitting when `brms` is installed.

The fit, diagnostics, posterior trajectory, and residual-scale model answer different questions. A successful fit is not itself evidence of adequacy, robustness, or substantive validity.

## Python API mapping

- `gp3bayespy.audit_pupil_temporal_dependence`
- `gp3bayespy.create_pupil_gp_spec`
- `gp3bayespy.diagnose_advanced_pupil_fit`
- `gp3bayespy.estimate_pupil_residual_scale`
- `gp3bayespy.fit_advanced_pupil_model_backend`
- `gp3bayespy.plot_advanced_pupil_simulation`
- `gp3bayespy.plot_advanced_pupil_trajectory`
- `gp3bayespy.plot_pupil_model_complexity`
- `gp3bayespy.plot_pupil_residual_scale`
- `gp3bayespy.plot_pupil_temporal_dependence`
- `gp3bayespy.predict_advanced_pupil_trajectory`
- `gp3bayespy.pupil_advanced_capabilities`
- `gp3bayespy.pupil_advanced_compatibility_table`
- `gp3bayespy.pupil_advanced_specification_table`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`
- `gp3bayespy.specify_pupil_distribution`
- `gp3bayespy.translate_advanced_pupil_model_to_brms`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
