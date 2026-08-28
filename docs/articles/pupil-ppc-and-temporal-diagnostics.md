# Pupil posterior predictive checks and temporal diagnostics

> Python-facing port of `pupil-ppc-and-temporal-diagnostics.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Posterior predictive evidence

Posterior predictive checks compare observed features with replicated data.
They are evidence objects, not automatic model-validity certificates.

The implementation summarizes observed and replicated trajectories, declared
window summaries, AUC, peak response and latency, residual structure, and
measurement-context overlays when corresponding indicators are available.

## Temporal residual review

Sampling diagnostics reuse the package's posterior/MCMC infrastructure and
report quantities such as R-hat, effective sample size, divergences,
treedepth, and available energy diagnostics. Temporal diagnostics additionally
show residual autocorrelation and support over event-relative time.

No single threshold is labelled proof of model adequacy. Measurement
limitations, specification uncertainty, and the prediction target remain
separate questions.

## Python API mapping

- `gp3bayespy.check_pupil_posterior_predictive`
- `gp3bayespy.diagnose_pupil_fit`
- `gp3bayespy.plot_pupil_ppc`
- `gp3bayespy.plot_pupil_residual_acf`
- `gp3bayespy.pupil_ppc_table`
- `gp3bayespy.pupil_residual_acf`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
