# Evidence Graphics and Governance

> Python-facing port of `evidence-graphics-and-governance.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The stable gp3bayes objects already preserve provenance, design support,
sensitivity results, backend checks, and schema comparisons. The development
graphics layer adds ggplot-based views while leaving the underlying objects
unchanged.

A status such as `pass`, `review`, or `fail` retains the semantics defined by
the originating gp3bayes audit. The plotting adapter does not reinterpret it.

## Python API mapping

- `gp3bayespy.backend_parity_table`
- `gp3bayespy.design_support_table`
- `gp3bayespy.manifest_comparison_table`
- `gp3bayespy.missingness_audit_table`
- `gp3bayespy.model_evidence_table`
- `gp3bayespy.plot_backend_parity_gg`
- `gp3bayespy.plot_design_support_gg`
- `gp3bayespy.plot_manifest_comparison_gg`
- `gp3bayespy.plot_missingness_gg`
- `gp3bayespy.plot_model_evidence_gg`
- `gp3bayespy.plot_schema_comparison_gg`
- `gp3bayespy.plot_sensitivity_suite_gg`
- `gp3bayespy.schema_comparison_table`
- `gp3bayespy.sensitivity_suite_table`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
