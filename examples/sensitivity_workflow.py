"""Declarative sensitivity planning without automatic model decisions."""

import gp3bayespy as gp

plan = gp.create_sensitivity_suite_plan(
    prior_scale=True,
    powerscale=True,
    psis_loo=True,
    prior_scale_args={"scale_multipliers": {"tighter": 0.5, "wider": 2.0}},
    powerscale_args={"prior_selection": None, "likelihood_selection": None},
)
print(plan)
print("Automatic model selection:", plan.automatic_model_selection)
print("Automatic exclusion:", plan.automatic_exclusion)
