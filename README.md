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

For AI explanations, set `GROQ_API_KEY` privately in the environment or ignored `.streamlit/secrets.toml`. See [AI setup](AI_SETUP.md) and [Deployment](DEPLOYMENT.md). The dashboard sends aggregate evidence to Groq; factual findings and planning actions remain available without online generation.

## Ask HarvestGuard

Click the floating chat icon at the bottom right of any page, type a question, and press Enter. The **Ask HarvestGuard** sidebar page also offers a larger view of the same conversation. Ask why a district ranks where it does, compare named districts, or ask how the data and model were prepared. The assistant retrieves matching aggregate results and methodology, checks numerical tokens and evidence references, and provides expandable sources, follow-up questions and an answer download. The current filters supply the default context; naming a different district, crop or season changes the question context. Chat history stays in the browser session and can be cleared.

Questions and relevant aggregate evidence are sent to Groq. Farmer-level inputs and credentials are never retrieved for answers. Local model comparisons are not causal explanations or measured intervention benefits.

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

 
