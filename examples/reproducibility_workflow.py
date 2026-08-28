"""Analysis manifest creation and deterministic provenance table."""

import gp3bayespy as gp

sim = gp.simulate_hierarchical_binary_data(
    n_participants=8, trials_per_participant=6, n_items=6, seed=31
)
contract = gp.create_model_contract(
    family="binary",
    outcome_col="selected",
    participant_col="participant_id",
    item_col="item_id",
    trial_col="trial_id",
    condition_col="condition",
)
prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
specification = gp.specify_binary_model(prepared)
manifest = gp.create_analysis_manifest(
    specification=specification,
    data=sim.data,
    seed=31,
    label="binary-demo",
)
print(gp.analysis_manifest_table(manifest))
