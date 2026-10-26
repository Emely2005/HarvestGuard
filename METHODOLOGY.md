# HarvestGuard Rwanda: data and methodology

HarvestGuard supports district planning for postharvest storage, processing and packaging. It combines survey loss observations, climate context and production quantities to show where estimated loss volumes are concentrated. The interface describes one selected trained model; recorded experiments remain available in the source exports.

## Sources and citations

- National Institute of Statistics of Rwanda (NISR). *Rwanda Seasonal Agricultural Survey 2024*, public-use version 0.1, reference **RWA-NISR-SAS-2024-v01**. [Catalogue, questionnaires and reports](https://microdata.statistics.gov.rw/index.php/catalog/113). Supplied DDI documentation: `ddi-documentation-english-113.pdf`.
- NISR. *Rwanda Seasonal Agricultural Survey 2025*, public-use version 0.1, reference **RWA-NISR-SAS-2025-v01**. [Catalogue, questionnaires and reports](https://microdata.statistics.gov.rw/index.php/catalog/124). Supplied DDI documentation: `ddi-documentation-english-124.pdf`.
- [CHIRPS](https://www.chc.ucsb.edu/data/chirps): daily gridded rainfall; the supplied preparation uses CHIRPS version 2 files for district rainfall averages.
- [NASA POWER](https://power.larc.nasa.gov/data-access-viewer/): daily RH2M relative humidity and T2M air temperature. Preparation uses inverse-distance interpolation to district centroids.
- [geoBoundaries Rwanda ADM2](https://www.geoboundaries.org/countryDownloads.html): administrative boundaries in the supplied climate preparation. The embedded dashboard geometry is retained.

The supplied survey PDFs describe national survey design. HarvestGuard's displayed loss quantities have their own scope: the scored sample, with no survey expansion weights applied.

## Cleaning and feature preparation

The original code and aggregate audit files are preserved in `source_preprocessing/`. They harmonise production and agricultural-practice modules for both years, decode Stata labels, normalise identifiers and category spellings, validate join cardinality, and aggregate repeated plot records to one farmer–crop–season–year observation. Repeated crop outcomes are checked for agreement before use. Negative loss values, conflicting repeated source quantities, missing loss components and actionable losses exceeding production invalidate the target rather than being guessed, clipped or changed to zero.

The archived preprocessing audit records 125,412 model-ready rows, including 66,104 in 2024 and 59,308 in 2025. The separate production-bearing inference file records 125,422 rows; the quantity export explicitly identifies and hashes that input. Dashboard coverage comes from that scoring run: **50,475 eligible 2025 profiles, 944 district–crop–season groups, 30 districts and 15 crops**. These are distinct recorded artifacts; their input row counts should not be treated as interchangeable.

Climate observations were joined by district and an estimated harvest-period midpoint, with a flagged season midpoint fallback. Preparation produced preharvest and historical postharvest windows. The selected model excludes realised postharvest rainfall, humidity and temperature; it uses only the preceding thirty-day rainfall and seven-day humidity windows. This prevents later realised weather from being presented as an input available at harvest.

The training pipeline validates finite targets between zero and one hundred, retains genuine zero outcomes, strips categorical text, checks harvest-month values, and converts invalid numeric inputs to missing values. One-hot encoding and numeric median imputation with missingness indicators are fitted within training partitions. Unseen categorical levels are handled at inference. Loss-component quantities, outcome flags, production quantities and record identifiers are excluded from predictive inputs; production is used after rate prediction.

## Target, model and validation

For profile *i*, the observed target is:

`loss_pct_i = 100 × (storage_loss_i + processing_loss_i + packaging_loss_i) / production_kg_i`.

The selected model is a two-part XGBoost model: a binary occurrence classifier and a Gamma regressor trained on positive losses. This matches the mixture of many zero-loss profiles and skewed positive outcomes:

`expected_loss_pct_i = P(loss_i > 0 | inputs_i) × E(loss_pct_i | loss_i > 0, inputs_i)`.

The ten inputs are season, district, crop, farmer type, harvest month, crop-plot count, total crop-plot area, crops per plot, preharvest rainfall and preharvest humidity. Training weights balance district–crop–season groups; these are model-fitting weights, not population expansion weights.

Development and selection use 2024 data. Three-fold stratified group cross-validation holds each district–crop–season group together. Separate grouped subsets within a training fold support early stopping; outer validation labels do not fit preprocessing or choose the stopping iteration. The saved tuning record contains twelve Optuna trials. The selected classifier and Gamma severity regressor use depth-three trees, at most five hundred boosting rounds, and early stopping. The locked configuration has no probability-calibration adjustment and no historical blend.

The selected configuration was subsequently evaluated on 2025. This is retrospective temporal validation, because that year had been inspected during project development.

| Recorded evaluation | Profile mean absolute error, percentage points | District loss-rate mean absolute error, percentage points |
| --- | ---: | ---: |
| 2024 grouped cross-validation | 0.4391498217 | 0.2164671715 |
| 2025 retrospective validation, full year | 0.4862533330 | 0.2801851826 |

Mean absolute error is `sum(abs(observed_pct - predicted_pct)) / number_of_observations`. These values come directly from `locked_choice_before_2025.json` and the selected model's `2025 all` row in `external_validation_2025.csv`. They measure percentage prediction errors, not classification accuracy, a competition score or the accuracy of the later kg ranking.

## Quantities and interpretation

Each profile is converted before aggregation:

`profile_loss_kg = expected_loss_pct / 100 × profile_production_kg`.

`district_loss_kg = sum(profile_loss_kg)`.

`district_loss_pct = 100 × district_loss_kg / sum(profile_production_kg)`.

District priority follows estimated kg lost for the selected crop and season. Multiplying an unweighted district mean percentage by total production would give a different and incorrect quantity calculation for these records. The saved model was reused without training or tuning for the quantity and explanation exports; model, input, quantity-table and importance hashes bind the displayed evidence together.

Predictive importance sums XGBoost `total_gain` across encoded columns belonging to an original feature, normalises within each model component, and averages occurrence and severity importance equally. Numeric reference comparisons replace one input at a time with its 2024 crop–season median, score the unchanged full model, and aggregate differences using production. They are predictive comparisons rather than causal or additive effects.

The explanation layer retrieves selected district evidence, peer loss-rate/volume contrasts, supported crops within the district and the same crop across historical seasons. It proposes only actions supported by those findings. Groq can select among those actions and explain the most relevant patterns in plain English. The interface renders its selected actions rather than always displaying the first two options. Evidence IDs, numerical placeholders, identity and supported actions are validated locally, with a bounded correction attempt. This is inference and retrieval, not online model learning.

## Limitations

Quantities cover sampled production; no whole-district or national loss totals are claimed. The dashboard is historical, not a live forecast. The kilogram ranking has no separately reported accuracy estimate. Season C has no reported positive-loss cases in the validation export. Global importance and reference substitutions do not establish causation, interventions' effects or savings. The combined target does not identify an individual loss stage, and ambient weather does not measure grain moisture or infrastructure. Small-group detail is suppressed, and point estimates have no calibrated uncertainty intervals. Costs, effectiveness and avoided losses require measurement in a field pilot.

## AI assistance

The project idea, problem framing and decisions belong to the HarvestGuard team. OpenAI Codex assisted with implementation, debugging, documentation and testing. Groq drafts explanations from selected aggregate evidence in the application. The team reviews the work, acknowledges sources and remains responsible for the submitted solution. Individual farmer records, private settings and API keys are excluded from the evidence packet.

**Ask HarvestGuard** answers open questions about the saved results and public methodology. The selected district, crop and season supply its default context; explicitly named filters retrieve matching groups, and follow-ups retain that context within the session. Its read-only retrieval uses validated aggregate exports and the checked-in methodology, selected model record and project audit. Each answer includes evidence references. Numerical tokens are checked against the cited facts before values and source links are inserted. Unverified responses receive at most one correction attempt. Chat history is bounded and session-local, rather than written into the shared narrative cache.

The current explanation service is Groq, with `openai/gpt-oss-120b` as the configured default language model. This assistant explains the findings; the statistical loss estimates still come from the unchanged selected two-part XGBoost model. It cannot execute actions, read private settings or farmer CSVs, or establish new causal effects. Structured-output and numerical checks do not prove that every natural-language inference is correct. Future-year forecasts, infrastructure conditions, measured intervention benefits and the unresolved original preparation steps remain outside the supplied evidence.
