# Publication-Ready Analysis Bundles

> Python-facing port of `publication-analysis-bundles.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The analysis-bundle layer assembles post-fit evidence while preserving the
distinction between computation, presentation, and interpretation.

No component is silently dropped: failures are retained in the bundle status
table with their error message.

## Explicit output

Reports and figures require explicit paths.

This design prevents analysis functions from writing automatically to the
working directory and keeps publication formatting downstream of the model
contract and validation objects.

## Python API mapping

- `gp3bayespy.analysis_bundle_table`
- `gp3bayespy.create_analysis_bundle`
- `gp3bayespy.create_analysis_figure_set`
- `gp3bayespy.create_publication_table_set`
- `gp3bayespy.save_figure_set`
- `gp3bayespy.write_analysis_bundle_report`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
