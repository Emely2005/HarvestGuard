# HarvestGuard Rwanda

HarvestGuard helps agricultural planners compare estimated postharvest loss volumes across Rwanda?s districts and turn the findings into seasonal service priorities for storage, processing and packaging.

The map ranks estimated **kilograms lost**. A two-part XGBoost model estimates each profile?s loss percentage; each percentage is multiplied by that profile?s production before aggregation. The district rate is production-weighted. The verified displayed coverage is **50,475 scored 2025 profiles, 30 districts, 15 crops and 944 district?crop?season groups**.

## Use the application

Choose a crop and season on **Map**, select a district, then open **Insights**. Compare district loss volumes, read the supported findings and actions, and download the ranking or explanation. **Metadata and Methodology** contains the source links and supplied survey PDFs, cleaning and training summary, formulas, selected-model evaluation, predictive-importance chart, limitations and AI disclosure.

The interpretation layer retrieves the selected district?s model evidence, volume-versus-rate contrasts, supported crop portfolio and historical seasonal patterns. Gemini can select among evidence-backed actions and explain the most useful findings. Its chosen actions are rendered on screen. Verified facts and planning actions remain available when online AI generation is unavailable.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run harvestguard_app.py
```

`streamlit run dashboard.py` opens the same application. Keep the application modules, `knowledge_base/`, `.streamlit/config.toml`, `model_v3_candidate/`, `model_v3_quantity_results/`, `model_v3_insights/`, the two survey PDFs and `source_preprocessing/` together. The web runtime reads aggregate exports and checks model provenance; it does not load or retrain the fitted model.

For AI explanations, configure `GEMINI_API_KEY` privately in the environment or ignored `.streamlit/secrets.toml`. The dashboard has no key-entry field. See [AI setup](AI_SETUP.md). Only aggregate evidence is sent; individual farmer records, private settings and credentials are excluded from the evidence packet. Transient requests use bounded backoff, failed selections do not restart on ordinary reruns, and **Retry AI** becomes available after its cooldown. No paid-provider fallback is configured.

## Data and methodology

The source surveys are NISR?s [Seasonal Agricultural Survey 2024](https://microdata.statistics.gov.rw/index.php/catalog/113) and [2025 survey](https://microdata.statistics.gov.rw/index.php/catalog/124). Climate preparation uses [CHIRPS rainfall](https://www.chc.ucsb.edu/data/chirps) and [NASA POWER humidity and temperature](https://power.larc.nasa.gov/data-access-viewer/). The displayed model uses preharvest rainfall and humidity; temperature is a preparation variable outside its inputs.

Read [Methodology](METHODOLOGY.md) for the original cleaning rules, source references, actual selected-model errors, equations and limitations. The supplied preprocessing code and aggregate audits are preserved in `source_preprocessing/`. Model experiments and original validation exports are retained for reproducibility; the application presents the selected model without version-comparison or readiness tables. [Model review](MODEL_REVIEW.md) retains the technical development record.

The displayed quantities cover sampled production; survey expansion weights were not supplied. Historical validation is retrospective, and kilogram-ranking accuracy and intervention effects have no separately reported estimates. The interface states these limits once in its limitations section rather than repeating them above the map or in each insight.

## Reproduce the analysis

[Model development notebook](HarvestGuard_Model_Development_V3_Colab.ipynb) contains the training pipeline. [Quantity aggregation notebook](HarvestGuard_Quantity_Aggregation_Colab.ipynb) reuses the saved model without fitting. `harvestguard_insights_export.py` reproduces the aggregate explanation export when the private production-bearing data and original model environment are available. The raw source data and profile predictions are excluded from the public release. `model_v3_quantity_results/RUN_RESULTS.md` and `model_v3_insights/RUN_RESULTS.md` record those completed runs.

## Tests and deployment

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python scripts/build_public_release.py
```

The release builder uses an explicit allow-list, removes notebook outputs in the release copy, checks for credential patterns and excludes private settings, caches and individual farmer data. The original local notebooks are retained. [Deployment instructions](DEPLOYMENT.md) give the Streamlit Community Cloud settings, private AI configuration and optional Docker deployment. Community Cloud uses the root `requirements.txt`, which includes the tested lightweight runtime from `requirements-web.txt`. A public repository and deployed URL require connected GitHub and Streamlit accounts; localhost is a local preview only.

[Submission brief](SUBMISSION.md) maps the demonstrated project to the five NISR evaluation criteria, provides a demonstration sequence and records AI assistance. The project idea, problem framing and decisions belong to the HarvestGuard team. OpenAI Codex assisted with implementation, debugging, documentation and testing; Google Gemini drafts explanations from aggregate evidence. The team reviews the work and remains responsible for the submitted solution.
