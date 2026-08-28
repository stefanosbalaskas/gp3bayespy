# Posterior pupil trajectories and declared estimands

> Python-facing port of `pupil-trajectories-and-estimands.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Lightweight posterior-draw contract

The estimand layer operates on posterior prediction draws. For documentation
and tests, a deterministic draw matrix can be used without compiling Stan.

## Trajectory uncertainty

Pointwise intervals describe uncertainty at each grid value. A finite-grid
simultaneous band can be requested explicitly; it is qualified as a grid-based
posterior band rather than a universal continuous-time guarantee.

## Declared contrasts and windows

Windows are supplied by the analyst. The package does not search across time
for the most favourable interval and relabel it confirmatory. Peak and
peak-latency summaries propagate posterior-draw uncertainty within the
declared evaluation grid.

## Python API mapping

- `gp3bayespy.as_pupil_prediction_draws`
- `gp3bayespy.estimate_pupil_auc`
- `gp3bayespy.estimate_pupil_peak`
- `gp3bayespy.estimate_pupil_peak_latency`
- `gp3bayespy.estimate_pupil_trajectory`
- `gp3bayespy.estimate_pupil_window`
- `gp3bayespy.pupil_condition_contrast`
- `gp3bayespy.pupil_trajectory_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
