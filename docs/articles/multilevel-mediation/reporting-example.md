# Worked reporting example

The example below shows how to report a **synthetic** trial-level mediation analysis without overstating the estimand. Replace every bracketed value with the output from the actual fitted model and retain the scale language appropriate to the selected families.

## Study-method wording

> We analyzed repeated trial-level observations using a hierarchical mediation model with participants as the clustering unit. The trial-varying exposure and gaze mediator were decomposed before modeling into within-participant deviations and participant means. The gaze mediator was modeled with a Gaussian likelihood and the binary override outcome with a Bernoulli-logit likelihood. Trials with missing outcomes or mediator values were not recoded as zero. The predeclared `quality_eligible` policy restricted the inferential dataset to rows meeting the gaze-quality rule while preserving excluded-row provenance. Participant random intercepts were included in both submodels, with a participant-varying exposure→mediator slope where supported by the design. Weakly informative priors were specified before sampling.

## Diagnostic wording

> Four chains were sampled for 1,000 post-warmup draws per chain after 1,000 warmup iterations using seed 2026. The largest R-hat was [1.00x], the minimum bulk ESS was [value], and the fit produced [0] divergences and [0] maximum-tree-depth hits. Posterior predictive checks were inspected for both the mediator and outcome submodels. Because all critical convergence checks passed, posterior indirect-effect summaries were extracted. [If any critical diagnostic fails, replace this sentence with a transparent failure statement and do not report the blocked indirect effect as valid.]

## Result wording for a binary outcome

> The within-participant coefficient-product indirect effect (`a_W × b_W`) was [posterior mean/median], with a 95% credible interval of [[lower], [upper]]. This estimand is on the model's linear-predictor product scale. It should not be interpreted as a percentage-point natural indirect effect on outcome probability. The direct within-participant path (`c'_W`) was [estimate, interval]. Between-participant exposure paths were [estimated / not estimable because `X_between` was constant in the balanced within-subject design].

## Interpretation wording

> The result is consistent with a trial-level pathway in which within-person variation in the exposure is associated with variation in the gaze mediator and, conditional on the exposure path, with the binary decision outcome. The hierarchical model separates this within-person pattern from stable between-person differences. Causal mediation language requires additional assumptions about temporal ordering and confounding; the coefficient-product estimate alone does not establish a natural indirect effect.

## Limitation wording

> The gaze mediator is an operational process measure and does not by itself establish the latent psychological construct often associated with that measure. In addition, the complete-case/quality-eligible estimand is conditional on the declared observation rule, and informative gaze loss could bias the fitted pathway. Sensitivity analyses compared [prior scales / likelihoods / missingness policies / random-slope structures], with [brief description of material or negligible changes].

## Compact results table

| Quantity | Posterior summary | Scale / note |
| --- | --- | --- |
| `a_W` | [estimate, 95% CrI] | mediator-model scale |
| `b_W` | [estimate, 95% CrI] | outcome linear-predictor scale |
| within indirect `a_W × b_W` | [estimate, 95% CrI] | linear-predictor product scale |
| within direct `c'_W` | [estimate, 95% CrI] | outcome linear-predictor scale |
| between indirect `a_B × b_B` | [estimate or not estimable] | report only when both paths are data-supported |
| max R-hat | [value] | convergence diagnostic |
| min bulk ESS | [value] | convergence diagnostic |
| divergences | [value] | should be reported explicitly |
| tree-depth hits | [value] | report when backend supplies it |

## Programmatic report helper

Use `report_multilevel_gaze_mediation()` for a structured model summary, but treat the returned text/table as a starting point for the manuscript rather than a substitute for design-specific interpretation. Effect extractors remain convergence-gated by default.

Related functions:

- `summarise_multilevel_mediation()`
- `estimate_within_indirect_effect()`
- `estimate_between_indirect_effect()`
- `posterior_direct_effect()`
- `posterior_total_effect()`
- `check_mediation_convergence()`
- `posterior_predictive_check_mediation()`
- `plot_indirect_effect_distribution()`
- `report_multilevel_gaze_mediation()`
