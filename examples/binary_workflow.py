"""Governed hierarchical-binary preparation and prior predictive checking."""

import gp3bayespy as gp

sim = gp.simulate_hierarchical_binary_data(
    n_participants=12, trials_per_participant=8, n_items=8, seed=7
)
contract = gp.create_model_contract(
    family="binary",
    outcome_col="selected",
    participant_col="participant_id",
    item_col="item_id",
    trial_col="trial_id",
    condition_col="condition",
    predictors=("participant_covariate", "trial_covariate"),
    interaction=("condition", "participant_covariate"),
    random_slope=True,
)
prepared = gp.prepare_hierarchical_binary_data(
    sim.data,
    contract,
    scale_predictors=("participant_covariate", "trial_covariate"),
)
specification = gp.specify_binary_model(prepared)
prior_check = gp.check_binary_prior_predictive(specification, draws=100, seed=7)
print(prepared.data.head())
print(prior_check.summaries)
