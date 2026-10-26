# Selected model and its original validation evidence

This is the model used to produce the deployed dashboard's estimates:
`harvestguard-v3-candidate-1`, selected trial 6. The original directory name and
artifact identifiers are retained so saved-model references and hashes continue
to match. The word `candidate` is part of that recorded identity.

`model_v3_candidate.joblib` is the original fitted model.
`harvestguard_model_v3.py` here is the source shipped with that saved model and
is required to load its Python classes. Keep it alongside the fitted artifact.
The root module with the same name adds the current inference and quantity
utilities; both notebooks embed that root module. These files have distinct
roles and must not be substituted by filename alone.

The CSVs, locked selection JSON and validation plot are evidence from this
model's development and retrospective evaluation, including baselines and
failed readiness checks. They are retained to make the evaluation reviewable.
`ranking_cohorts_2025.csv` retains the data that was also exported byte for byte
as `ranking_detail_candidate.csv`; that redundant copy was removed.

The current map reads the kg priorities in `../model_v3_quantity_results/`.
`harvestguard_2025_v3_vulnerability.csv` is the original rate-only export used
for provenance and regression checks, not a competing current map dataset.
