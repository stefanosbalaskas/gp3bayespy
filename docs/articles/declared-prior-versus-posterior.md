# Declared Priors versus Fitted Posteriors

> Python-facing port of `declared-prior-versus-posterior.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The bridge uses the backend-independent prior specification retained by
`gp3bayespy`; saved backend prior draws are not required.

Shift, contraction, interval overlap, Kolmogorov-Smirnov distance, and
quantile-based Wasserstein distance are descriptive marginal summaries, not
automatic measures of prior adequacy.

## Python API mapping

- `gp3bayespy.plot_prior_posterior_contraction`
- `gp3bayespy.plot_prior_posterior_density`
- `gp3bayespy.plot_prior_posterior_intervals`
- `gp3bayespy.plot_prior_posterior_shift`
- `gp3bayespy.prior_posterior_bridge`
- `gp3bayespy.prior_posterior_distance_table`
- `gp3bayespy.prior_posterior_summary_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
