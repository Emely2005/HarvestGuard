RWANDA POST-HARVEST RISK — CORRECTED MODEL-READY PREPROCESSING

FINAL MODELING GRAIN
One row = farmer x crop x season x year.

REGRESSION TARGET
actionable_loss_rate_pct
= 100 * (storage loss + processing loss + packaging loss) / total crop production

IMPORTANT TARGET CLEANING RULES
1. Negative storage/processing/packaging loss values are treated as source anomalies.
   They are NOT converted to positive values and NOT converted to zero.
2. If a farmer-crop group contains any negative actionable-loss source value, that
   observation is excluded from regression target training.
3. If repeated plot rows disagree on total crop production or on storage/processing/
   packaging loss, the farmer-crop observation is excluded rather than silently using max().
4. If actionable loss exceeds total reported production, the target is excluded.
5. Missing target components are not filled with zero.
6. Transport and sales loss are not part of the regression target.

FINAL X FEATURES
season
district
crop_name
farmer_type
harvest_month
n_plots_crop
crop_plot_area_ha_total
num_crops_in_plot
rain_30d_before_mm
rain_14d_after_mm
rainy_days_14d_after
humidity_mean_7d_before_pct
humidity_mean_14d_after_pct
temperature_mean_14d_after_c

SPLIT
year is included ONLY for train/test splitting:
- train: 2024
- test: 2025
Do not include year in X.

MULTI-RESPONSE QUESTIONNAIRE FIELDS
Detailed irrigation/fertilizer/pesticide multi-response fields are intentionally NOT in the
final X feature set, so they are not collapsed into artificial single categories for model
training. If such fields are later added as predictors, all response columns must first be
combined according to the questionnaire structure.

FINAL RESULT
125,412 model-ready rows
14 X features
1 continuous regression target
0 missing values in selected X/y
30 districts
