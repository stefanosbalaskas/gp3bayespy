# Gaussian-Process Pupil Trajectories

> Python-facing port of `gaussian-process-pupil-trajectories.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

# Approximate GP is the default

Exact GP computation remains available, but the complexity audit requires explicit review when the unique time-by-condition grid becomes large.

# Hyperparameters are posterior estimands

Length scale and marginal GP standard deviation describe the fitted temporal function prior/posterior. They are not direct psychological constructs.

## Python API mapping

- `gp3bayespy.create_pupil_gp_spec`
- `gp3bayespy.fit_advanced_pupil_model_backend`
- `gp3bayespy.plot_advanced_pupil_trajectory`
- `gp3bayespy.plot_pupil_gp_hyperparameters`
- `gp3bayespy.plot_pupil_model_complexity`
- `gp3bayespy.predict_advanced_pupil_trajectory`
- `gp3bayespy.pupil_gp_hyperparameters`
- `gp3bayespy.pupil_gp_table`
- `gp3bayespy.simulate_advanced_pupil_timecourse`
- `gp3bayespy.specify_advanced_pupil_timecourse_model`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
