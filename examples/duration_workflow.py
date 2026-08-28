"""Governed hierarchical-duration preparation and prior predictive checking."""

import gp3bayespy as gp

sim = gp.simulate_hierarchical_duration_data(
    n_participants=12, trials_per_participant=8, n_items=8, seed=8
)
contract = gp.create_model_contract(
    family="duration",
    outcome_col="duration",
    participant_col="participant_id",
    item_col="item_id",
    trial_col="trial_id",
    condition_col="condition",
    predictors=("participant_covariate", "trial_covariate"),
    interaction=("condition", "participant_covariate"),
    random_slope=True,
    outcome_unit="milliseconds",
)
prepared = gp.prepare_hierarchical_duration_data(
    sim.data,
    contract,
    scale_predictors=("participant_covariate", "trial_covariate"),
)
specification = gp.specify_duration_model(prepared, baseline=500.0)
prior_check = gp.check_duration_prior_predictive(specification, draws=100, seed=8)
print(prepared.data.head())
print(prior_check.summaries)
