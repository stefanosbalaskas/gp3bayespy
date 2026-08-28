# Simulation-Based Calibration Diagnostics

> Python-facing port of `sbc-publication-diagnostics.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The existing SBC workflow delegates computation to `SBC` and retains a
conservative gp3bayes result.

Graphical agreement is not converted into an automatic implementation-validity
claim.

## Python API mapping

- `gp3bayespy.plot_sbc_coverage_gg`
- `gp3bayespy.plot_sbc_ecdf_gg`
- `gp3bayespy.plot_sbc_rank_gg`
- `gp3bayespy.plot_sbc_simulated_vs_estimated_gg`
- `gp3bayespy.sbc_overview_table`
- `gp3bayespy.sbc_stats_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
