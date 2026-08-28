# Publication Registries and Diagnostic Dashboards

> Python-facing port of `publication-registries-and-dashboards.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

Registries bind named tables and figures to captions and provenance labels.
They write nothing unless an explicit output path is supplied.

Dashboards are non-interactive evidence indices and do not launch expensive
analyses implicitly.

## Python API mapping

- `gp3bayespy.create_diagnostic_dashboard`
- `gp3bayespy.create_diagnostic_dashboard_figures`
- `gp3bayespy.create_publication_registry`
- `gp3bayespy.diagnostic_dashboard_table`
- `gp3bayespy.plot_diagnostic_dashboard`
- `gp3bayespy.plot_posterior_intervals`
- `gp3bayespy.posterior_interval_table`
- `gp3bayespy.publication_registry_table`
- `gp3bayespy.register_publication_figure`
- `gp3bayespy.register_publication_table`
- `gp3bayespy.validate_publication_registry`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
