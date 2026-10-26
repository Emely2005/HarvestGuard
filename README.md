# HarvestGuard Rwanda

HarvestGuard helps agricultural planners compare estimated postharvest loss volumes across Rwanda's districts and plan seasonal storage, processing and packaging services.

The deployed dashboard uses one saved model: **harvestguard-v3-candidate-1, trial 6**. Its two-part XGBoost model predicts loss percentages. Each profile's prediction is multiplied by that profile's production before district aggregation. The map ranks **estimated kilograms lost**, covering **50,475 sampled 2025 profiles, 30 districts, 15 crops and 944 district/crop/season groups**.

## Start here

1. Read [Submission brief](SUBMISSION.md) for the problem, demonstration and AI disclosure.
2. Read [Methodology](METHODOLOGY.md) for sources, cleaning rules, validation and limitations.
3. Read [Project audit](PROJECT_AUDIT.md) for retained files, removed duplicates and the remaining reproduction gaps.
4. Read [Selected model record](MODEL_REVIEW.md) for the model and validation evidence used by this application.

## Run the dashboard

```powershell
python -m pip install -r requirements.txt
python -m streamlit run harvestguard_app.py
```

`dashboard.py` is a compatibility entry point for the same application. Choose crop and season on Map, select a district, then open Insights. Metadata and Methodology contains survey PDFs, source references, formulas, model errors, predictive importance and limitations.

For AI explanations, set `GEMINI_API_KEY` privately in the environment or ignored `.streamlit/secrets.toml`. See [AI setup](AI_SETUP.md) and [Deployment](DEPLOYMENT.md). The dashboard sends aggregate evidence to Gemini; factual findings and planning actions remain available without online generation.

## Follow the data and code

| Step | Authoritative files | Role |
| --- | --- | --- |
| Clean and unify survey/climate inputs | `source_preprocessing/build_postharvest_model_ready_corrected.py` and its requirements, dictionary, manifest and quality reports | Preserved cleaning implementation and aggregate audit |
| Develop the model | `HarvestGuard_Model_Development_V3_Colab.ipynb`, root `harvestguard_model_v3.py` | Training, grouped validation, selection and evaluation code |
| Reuse the fitted model | `model_v3_candidate/` | Selected saved artifact, original loading module, environment pins and original evaluation evidence |
| Convert profile rates to kg and aggregate | `HarvestGuard_Quantity_Aggregation_Colab.ipynb` | Inference with the existing model; no fitting |
| Supply the map | `model_v3_quantity_results/` | Current district quantity priorities and provenance |
| Produce and display explanations | `harvestguard_insights_export.py`, `model_v3_insights/`, `knowledge_base/` | Aggregate model evidence and supported planning actions |
| Run and deploy | `harvestguard_app.py`, supporting `harvestguard_*.py`, `.streamlit/`, requirements and hosting configuration | Dashboard and deployment |
| Verify and package | `tests/`, `model_review_tests/`, `scripts/` | Regression checks, file audit and explicit public-release allow-list |

The root model module and the module inside the saved bundle have different roles; see [bundle explanation](model_v3_candidate/README.md). Historical V2 aggregate inputs survive only as clearly labelled regression fixtures in `tests/fixtures/legacy_v2/`.

The source surveys are NISR's [2024](https://microdata.statistics.gov.rw/index.php/catalog/113) and [2025](https://microdata.statistics.gov.rw/index.php/catalog/124) Seasonal Agricultural Surveys. Climate preparation uses [CHIRPS rainfall](https://www.chc.ucsb.edu/data/chirps) and [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/) humidity and temperature. The selected model uses preharvest rainfall and humidity; temperature is outside its inputs.

Raw survey archives, the private unified CSVs and individual profile predictions belong in the private working collection. They are excluded from GitHub. **The original 2025 pre-merge code and exact production-bearing export step are still missing from this project.** The recorded cleaning output has 125,412 rows; the inference input has 125,422. The audit explains this unresolved difference. The existing training and aggregation code must not be presented as complete raw-to-deployment reproduction until these steps are recovered.

## Check the submission

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/audit_project.py --verify-manifest
python scripts/build_public_release.py
```

The audit reports missing required files and unresolved source steps. Add `--data-dir PATH --raw-dir PATH --require-reproducibility` to inspect private inputs and return a failure when the full reproduction chain is incomplete. See [Project audit](PROJECT_AUDIT.md).

The release builder includes only explicitly named files, clears notebook outputs in the release copy, checks for credential patterns and produces a checksummed manifest. Historical archives, caches, private settings and individual records are excluded. The dashboard reads aggregate exports; it does not fit or load the training model, but checks its provenance.

The displayed quantities cover sampled production; survey expansion weights were not supplied. Validation is retrospective. Separate kg-ranking accuracy and intervention benefits have not been established.

The project idea, problem framing and decisions belong to the HarvestGuard team. OpenAI Codex assisted implementation, debugging, documentation and testing. Google Gemini drafts explanations from aggregate evidence. The team reviews the work and remains responsible for the submission.
