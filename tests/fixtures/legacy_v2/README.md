# Historical regression-test inputs

These two aggregate V2 files are used by `tests/test_ai.py` and
`tests/test_decisions.py` to check percentage conversion, evidence validation and
the earlier rate-only interface. They are test fixtures, not the deployed model
or current map data. Keeping them preserves meaningful regression coverage.

The application uses `model_v3_candidate/`, `model_v3_quantity_results/` and
`model_v3_insights/`. Current-model integration tests use those directories.
