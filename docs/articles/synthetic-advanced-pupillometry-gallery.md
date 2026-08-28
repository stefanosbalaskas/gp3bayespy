# Synthetic Advanced Pupillometry Gallery

> Python-facing port of `synthetic-advanced-pupillometry-gallery.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Observed trajectories

# Stored latent mean

# Temporal dependence

# Missingness

# Computational audit

# Binocular example

The gallery is intentionally backend-free so it can be built during package checks without compiling Stan. Full posterior plot examples are shown in the modelling vignettes with fitting code disabled by default.

## Python API mapping

- `gp3bayespy.audit_binocular_pupil_readiness`
- `gp3bayespy.audit_pupil_measurement_model`
- `gp3bayespy.audit_pupil_missingness`
- `gp3bayespy.audit_pupil_temporal_dependence`
- `gp3bayespy.create_pupil_measurement_model`
- `gp3bayespy.create_pupil_missingness_spec`
- `gp3bayespy.plot_advanced_pupil_simulation`
- `gp3bayespy.plot_pupil_measurement_uncertainty`
- `gp3bayespy.plot_pupil_missingness`
- `gp3bayespy.plot_pupil_model_complexity`
- `gp3bayespy.plot_pupil_temporal_dependence`
- `gp3bayespy.prepare_binocular_pupil_timecourse`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.simulate_binocular_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
