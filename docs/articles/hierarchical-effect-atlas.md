# Hierarchical Effect and Variance Atlases

> Python-facing port of `hierarchical-effect-atlas.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The expanded group-level layer can inspect raw posterior deviations, rank
uncertainty, and baseline latent variance partitioning.

Rank probabilities and variance fractions remain descriptive posterior
quantities; they do not establish substantive group importance or causal
variance attribution.

## Python API mapping

- `gp3bayespy.group_effect_draws_table`
- `gp3bayespy.group_effect_rank_probability_table`
- `gp3bayespy.plot_group_effect_distribution`
- `gp3bayespy.plot_group_effect_rank_probability`
- `gp3bayespy.plot_random_intercept_variance_partition`
- `gp3bayespy.random_intercept_variance_partition`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
