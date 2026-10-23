# HarvestGuard model review and Colab next steps

## Insights and intervention planning (22 October 2026)

The revised Insights page keeps the existing header and KPIs, then shows the meaning, kg-based priority, model findings and recommended actions. Technical information is grouped under **Data, sources & privacy** and **Model metadata & validation**.

`harvestguard_insights_export.py` has run on the real production-bearing input using the unchanged trained candidate. It reproduced all 944 saved group quantities from 50,475 profiles before exporting district evidence. No fitting or tuning is performed. Outputs live in `model_v3_insights/`, separate from the original candidate and quantity files.

Global importance is original-feature summed XGBoost `total_gain`, normalized separately for occurrence and severity, then averaged equally. Crop, crop plot area and district rank highest; rainfall is fourth and humidity fifth. This describes this fitted model, not causal importance. Local numeric sensitivity replaces one input at a time with its 2024 crop-season median, re-scores the complete calibrated model and aggregates differences using production. Differences are model comparisons, not SHAP decompositions, additive quantities, intervention effects or expected savings. Reference substitutions can break correlations. The export documents the fallback reference, method, input/model hashes and group support.

For Nyagatare maize, Season A, the largest-production 20.2% of sampled profiles account for 98.3% of estimated loss. The January harvest cohort accounts for 60.4%. Its humidity is below the district comparison median, so it does not receive a humidity-driven service recommendation. The selected actions prioritize high-volume batches and align service capacity with that harvest cohort. The month describes the cohort, not the observed date of loss.

Moisture-control services are selected only for supported grains when measured preharvest weather exceeds the same-crop/season district median, weather coverage is at least 90%, and a positive local sensitivity is among the three largest numeric sensitivities. This is an explicit planning rule, not a validated treatment-effect model. Humidity does not identify a cold district; temperature is absent from this candidate. No facility shortage, storage condition, grain moisture measurement or equipment benefit is inferred. Other crops receive production and timing actions when the relevant aggregate evidence exists. Detail is suppressed for small groups. AI recommendations must cite both the intervention record and its required district findings.

Research basis: [XGBoost's prediction documentation](https://xgboost.readthedocs.io/en/stable/prediction.html) distinguishes prediction quantities and raw model contributions; [SHAP's methodological guidance](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html) explains why predictive attribution does not establish policy effects. [FAO's storage fundamentals](https://www.fao.org/4/x5065e/x5065e04.htm) supports the moisture-control practice link for grains. That source does not establish local infrastructure needs or benefits; the local selection uses our aggregate data and saved model comparisons.

## Updated intervention priority: kilograms lost

### Production-bearing input now supplied (22 October 2026)

`postharvest_regression_model_ready_2024_2025_with_production.csv` contains **125,422 rows and 17 columns**, including `total_crop_production`. The user confirmed that this column is in **kg**. It has no missing, non-finite, zero or negative values. All saved model inputs are present. Its 2025 eligible cohort still has **50,475 profiles and 944 district/crop/season groups**, matching every saved group count. The ten additional records in the complete file do not change this eligible 2025 cohort. Shared numeric inputs have small serialization differences, so the saved model is applied directly to the new file's rows instead of attaching predictions by row position.

The aggregation notebook is now configured for this filename, `PRODUCTION_COLUMN='total_crop_production'` and `PRODUCTION_UNIT='kg'`. It applies the existing model, converts each profile's rate to kg using its own production, then sums those profile losses. Its checks verify both the row conversion and the group sums. Profile results retain the individual kg values; no rounding occurs before aggregation. The output manifest records the input/model hashes and that the source model file remained unchanged. No survey expansion weight column is present, so these quantities are sample totals.

The shipped aggregation notebook has now been executed on the **real production-bearing CSV**, using the saved V3 candidate with fitting/tuning calls blocked in the verification run. It scored 50,475 profiles into 944 groups and saved results in `model_v3_quantity_results/`. Per-profile conversion and group sums passed, and the source model hash was unchanged. Recomputed unweighted group mean predictions differ from the original saved predictions by at most **8.90 × 10⁻¹⁰ percentage points**. This confirms inference consistency, not new predictive accuracy. `real_data_aggregation_checks.json` records the checks.

The app is now connected to the verified quantity folder. Its map, Insights, briefs and ranking downloads use estimated kg lost. The loader checks the quantity manifest against the original model's hash, candidate version, year and features and rejects partial or inconsistent results. Original percentage-validation files stay in the model bundle and are explicitly labelled as earlier rate-based evidence; quantity-ranking accuracy has not been re-evaluated. Updating the quantity CSV or metadata invalidates the app's data cache.

Local Windows loading exposed the [documented XGBoost cross-platform pickle RNG issue](https://github.com/dmlc/xgboost/issues/12459). The inference loader uses the original snapshot's unmodified Model byte section through XGBoost's native model loader for this specific error, leaving the trees, preprocessing and calibration intact. It does not restore training RNG state or retrain. The notebook uses ordinary joblib loading when it works, including the original Colab environment. See [XGBoost's model IO documentation](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html) for model files versus training snapshots.

### Inspection of the downloaded input CSV (22 October 2026)

`postharvest_regression_model_ready_2024_2025.csv` is now in the workspace. It contains **125,412 rows and 16 columns**, covering 2024 and 2025, with all saved V3 model features present and no missing cells. There are 59,308 rows for 2025; 50,475 belong to the model's eligible crops. Those eligible rows reproduce all **944** saved district/crop/season groups: profile counts match exactly and mean observed loss percentages agree within floating-point precision. This checks consistency of the supplied rows and aggregate outcomes; it does not establish raw-survey provenance or validate quantity-based priorities.

The CSV contains **no production quantity, stage loss quantity, survey expansion weight or farmer/profile identifier**. `actionable_loss_rate_pct` is an observed percentage target, not kg. `crop_plot_area_ha_total` is area in hectares, not production. Consequently there is no valid `PRODUCTION_COLUMN` to configure in this file, and changing its name or retraining cannot recover kg totals.

This describes the earlier 16-column file only. The later production-bearing input above supplies the missing denominator and is the configured input for the aggregation notebook. No kg-loss totals are inferred from the older file.

For the aggregation-only change, use [HarvestGuard_Quantity_Aggregation_Colab.ipynb](HarvestGuard_Quantity_Aggregation_Colab.ipynb). It loads the existing trained V3 model, scores the selected year's eligible profiles and aggregates kg. It never calls training or tuning. The original bundle stays unchanged; a separate ZIP contains quantity results and provenance for review. The development notebook below is only for an intentional new training/selection experiment.

For the supplied production-bearing file, `PRODUCTION_COLUMN` is explicitly `total_crop_production`. If a different input fails at production mapping, run `print(raw.columns.tolist())` and configure its exact denominator name and unit. The optional `SOURCE_QUANTITIES` target audit is separate: its informational skipped-audit message does not cause a missing-production error. If no compatible quantity column exists in another input, it must be restored from the original data; neither retraining nor an invented default can recover it.

The rate model still predicts each farmer/crop/season profile's combined storage, processing and packaging loss percentage. The revised post-prediction calculation converts **each profile first**, then aggregates:

```text
profile predicted loss kg = predicted loss percentage / 100 × profile production kg
district/crop/season predicted loss kg = sum of profile predicted loss kg
priority rank = descending district/crop/season predicted loss kg
district loss percentage = 100 × summed loss kg / summed production kg
loss tonnes = loss kg / 1,000
```

For illustration, Nyarugenge losing 30% of 1,000 kg means 300 kg lost; Nyagatare losing 4% of 100,000 kg means 4,000 kg lost. Nyagatare ranks first by quantity. These quantities illustrate the rule and are not measured district results. Multiplying an unweighted average percentage by total production is incorrect when farmers have different production volumes.

In the Colab configuration, map `PRODUCTION_COLUMN` to the original profile production denominator and set `PRODUCTION_UNIT` to `kg` or metric `tonnes`. The notebook lists the actual CSV columns before mapping and stops if production is missing, incomplete, nonpositive or invalid. Restore the quantity by a verified profile/crop/season/year join if cleaning removed it; never join by row position or repeat one district total on every farmer. Production stays outside the predictive feature list. The realized production used in historical evaluation must be replaced with available or explicitly forecast production for a future decision.

With verified `SURVEY_WEIGHT_COLUMN` expansion weights, both production and lost kg are multiplied by the profile's weight before summing. Without them, exports say `sample_total`: these are totals for the sampled profiles, not population district totals. With weights they say `survey_expanded`, still restricted to the covered profiles and eligible crops. Model training weights must never be substituted for survey expansion weights.

The revised pipeline selects and evaluates district rankings using **observed and predicted kg lost**, with a district MAE guardrail in kg. Profile percentage errors and production-weighted district percentage errors remain diagnostics. Candidate and historical baselines use the same production, weights and evaluation rows. The model target remains a percentage; this is a change to aggregation and the priority objective. Previously exported rate-ranking scores below do not validate quantity-based ranking.

The dashboard supports quantity exports in the existing V3 CSV filename: `production_kg`, `predicted_loss_kg`, `observed_loss_kg`, `predicted_loss_tonnes`, `observed_loss_tonnes`, `priority_rank` and `loss_quantity_scope`. Its map, ranking CSV, briefs and AI evidence use kg priorities when available; ties share ranks. Metadata records the basis and quantity scope. `predicted_loss_pct` and `observed_loss_pct` are production-weighted in these exports. Unweighted profile means remain separately named diagnostic columns.

**The original saved V3 bundle remains unchanged.** The app combines it with the verified separate quantity CSV/metadata and displays kg priorities. If quantity results are absent, its legacy rate-only view is explicitly labelled. Quantity results supplement the model bundle; they do not replace the trained artifact or establish quantity-ranking accuracy. No district kg totals have been inferred from the older district mean percentages.

## Earlier review of loss-rate models

The main problem is generalization of district rankings. The historical baseline actually ranks 2025 districts better than the current Hurdle XGBoost model. More trees or a different AI explanation cannot fix that by themselves.

The dashboard currently displays saved district predictions from a CSV. It checks that model artifacts exist but does not run XGBoost inference when someone changes a filter.

The saved JSON artifacts are consistent with a genuine two-stage Hurdle XGBoost: a 294-tree binary-logistic occurrence model and a 53-tree Tweedie severity model, each expecting 73 encoded features. That checks their basic structure, not their accuracy or the provenance of the training rows.

## What the supplied results show

Source: `model_v2/external_validation_2025.csv`, **2025 Seasons A+B**, using V2's original evaluation rules.

| Metric | Hurdle XGBoost | Historical baseline | Better direction |
|---|---:|---:|---|
| Profile MAE, percentage points | 0.4976 | 0.5003 | Lower |
| Macro district NDCG@5 | 0.1996 | 0.2580 | Higher |
| Top-five overlap | 0.2267 | 0.2667 | Higher |

The MAE improvement is only about **0.55% relative**. The ranking metrics favor the baseline. Hurdle R² is about **−0.036**: that is not evidence of strong profile-level predictive accuracy. A negative R² means squared error exceeds the constant evaluation-mean reference on those rows; that mean is a diagnostic reference, not a deployable forecast. See [scikit-learn's R² definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).

The reviewed notebook's hard-coded “verified reference results” paragraph gives different figures from these saved exports. Use generated result tables for the competition, not that stale paragraph.

## Findings and changes

1. **Zero-heavy outcomes make one score misleading.** The eligibility export contains 56,484 development profiles and 3,997 positive losses: **92.92% are zero**. From the saved 2025 A+B calibration bins, predicting zero everywhere would have approximately 0.2722 pp MAE, lower than the model's 0.4976. That would still be a poor expected-loss model and district decision tool. Evaluate RMSE, expected-rate calibration, ranking and occurrence AP/Brier together. Do not describe the system as “93% accurate.”

2. **The strongest predictions are poorly calibrated in the next year.** In the saved 2025 highest-score decile, average predicted loss is **1.1522%**, while average observed loss is **0.2484%**. Middle bins have higher observed loss than that top bin. This points to poor discrimination/calibration; the exports alone cannot prove its cause. V3 tests held-out 2024 calibration and separate occurrence/severity diagnostics. Do not fit a correction to 2025 outcomes and reuse 2025 as a blind test.

3. **The tuning budget never reached adaptive TPE.** V2 uses 4 hurdle trials, 4 CatBoost trials and 5 XGBoost trials. `TPESampler` defaults to 10 random startup trials, so those searches remained in the startup phase. V3 explicitly uses a smaller startup budget with 12 initial trials; more trials can follow after auditing the data. See [Optuna's startup rule](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

4. **Early stopping uses the fold that is also scored.** That adds selection optimism to V2 CV results. V3 chooses the tree count using an inner training-only subset, reserves separate training-only calibration rows, then scores the outer fold. Hyperparameter selection still reuses OOF results, so those are selection scores, not an unbiased nested-CV claim. The later temporal evaluation is essential.

5. **The CV baseline is weaker than the temporal baseline.** V2 CV uses crop/season means; 2025 uses exact prior-year district/crop/season means when available. Holding out an entire district/crop/season group removes its exact history in CV, unlike many 2025 predictions. V3 includes exact and smoothed hierarchical baselines and baseline/ML blends. With only one development year, it still cannot perfectly simulate repeated same-season year-to-year transfer. Additional independent years would address this limitation more directly than an enormous parameter search.

6. **Occurrence and severity need different settings.** V2 shares depth and `min_child_weight` between a classifier on all profiles and a severity model on only positive profiles. V3 separates them and tests Gamma versus Tweedie positive-severity models in percentage-point units. These are candidates, not guaranteed improvements. XGBoost defines minimum child weight through accumulated Hessians, so the same number need not imply similar regularization for different objectives. See [XGBoost's parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).

7. **Rare positive cases and source definitions limit what can be learned.** Groundnut has only 16 positive 2024 cases, Pea 11, and Sunflower 12. Complex crop-specific models are not justified by large zero counts. The 2025 Season C export has zero positive losses; verify survey skip rules, missing-value codes, units and cleaning before assuming those are all genuine zeros. The supplied cleaned-data notebook cannot reconstruct the target arithmetic because production and stage quantities were already removed.

8. **Weather availability must match the prediction moment.** V2 includes realized weather after harvest. That is valid for a clearly labeled retrospective experiment, but does not establish an operational forecast. V3 defaults to past-weather inputs only. Set `FEATURE_MODE='retrospective'` for an explicitly historical experiment; never change the label to “live forecast.” Even the past-weather joins and planned harvest month need timestamp/provenance checks.

The train-versus-CV MAE values are almost equal, so these exports do **not** support a simple diagnosis of severe conventional training overfit. Weak signal, rare positives, aggregation, temporal shifts and data construction remain important hypotheses to investigate.

## Optional: train and select a new candidate

Open [HarvestGuard_Model_Development_V3_Colab.ipynb](HarvestGuard_Model_Development_V3_Colab.ipynb) in Google Colab using **File → Upload notebook**. It includes its modeling code; no companion file upload is needed.

1. Keep your model-ready CSV at `/content/drive/MyDrive/NISR/postharvest_regression_model_ready_2024_2025.csv`, or change `DATA_PATH`.
2. Check `FEATURE_MODE`, the required production column/unit, optional verified survey expansion weights, optional entity-ID column and target-quantity audit mapping. Do not guess missing IDs or original quantity columns.
3. Run all cells. Start with 12 trials; inspect evidence before increasing to 30–40. CPU is sufficient for the implementation. Checkpoints save completed trials and resume when data, source, environment and configuration match.
4. Read the candidate versus baseline comparison on the **same 2025 rows, production, weights and cohort rules**. The revised rankings evaluate kg totals. Do not compare their scores directly with earlier rate-based evaluations.
5. Download `model_v3_retrained_quantity.zip` (or the filename matching your chosen output folder). The notebook verifies save/reload consistency and demonstrates inference on feature rows without loss labels.

The candidate can be a historical baseline, single-stage model, hurdle model or blend. It is not automatically called Hurdle XGBoost. The app reads the saved V3 bundle; a new Colab run does not update that local bundle automatically.

## What to send back after your run

Share these summary outputs; no API key or raw farmer records are needed:

- `external_validation_2025.csv`
- `external_validation_by_crop_2025.csv`
- `component_diagnostics_2025.csv`
- `target_by_year_season.csv`
- `model_v3_metadata.json`

Prefer the simpler baseline if the candidate does not improve district ranking and maintain useful error/calibration. Point-metric gains alone do not establish statistical reliability, intervention effectiveness or national allocation readiness. More independent years, design-aware uncertainty, compatible production quantities, verified survey weights and genuinely available storage/handling measurements are the strongest longer-term upgrades.

## Verification and limits

The revised module is tested separately from the app. Tests check target bounds, feature timing, group separation, held-out-label independence, fair ties, saved-model inference, candidate selection and exports. The notebook is also executed on synthetic data to check the complete workflow. Synthetic scores are not Rwanda performance results.

Optional local model checks use the dependencies in `model_review_tests/requirements.txt`, then `python -m pytest model_review_tests -q`. The Colab notebook installs its own dependencies.

Both model-ready CSVs are now in the workspace. The newer file supplies the production denominator, and the aggregation notebook has been executed on its real 2025 rows with the existing trained model. Original stage loss quantities are still absent, so an independent reconstruction of the observed target from storage/processing/packaging quantities remains unresolved. No real-data refitting or improved-performance claim was made. A new uninspected evaluation period is still needed after the already-inspected 2025 results.

Calibration method reference: [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html). The V3 calibrator uses held-out training rows rather than predictions from models fitted on those calibration labels.
