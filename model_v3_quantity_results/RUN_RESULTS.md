# HarvestGuard aggregation run results

Run date: 22 October 2026.

The saved V3 model was reused. No training, tuning or calibration was performed.

Input: `postharvest_regression_model_ready_2024_2025_with_production.csv` (125,422 rows across 2024 and 2025).
Scored: 50,475 eligible 2025 profiles; 8,839 2025 profiles excluded because their crops are outside saved model eligibility.
Output: 944 season/crop/district groups, covering 30 districts and 15 crops.

## Verified calculation

For each profile: predicted loss kg = predicted loss percentage / 100 * total_crop_production (user-confirmed kg).
Then sum those individual loss quantities by season, crop and district. No rounding occurs before aggregation.
District percentage = 100 * summed loss kg / summed production kg. Rank uses summed loss kg.

Per-profile conversion and group sums passed. The saved model file is unchanged.
All 944 district priority ranks agree with the uploaded Colab output. Small floating-point differences in kg do not change the ranks.

## Example: maize, Season A

| District | Estimated kg lost | Priority rank |
|---|---:|---:|
| Nyagatare | 64,545.97 | 1 |
| Kirehe | 17,803.26 | 2 |
| Kamonyi | 11,782.17 | 3 |
| Nyarugenge | 39.98 | 26 |

## Coverage

Quantities cover sampled profiles only. No survey expansion weights were supplied; these are not whole-district or national population totals.
This aggregation run does not establish new ranking accuracy or intervention benefits.

## Files

- `profile_quantity_predictions_2025.csv`: individual profile predictions and kg losses.
- `harvestguard_2025_v3_quantity_priorities.csv`: district/crop/season quantities and ranks.
- `quantity_aggregation_metadata.json`: source identity, input/model hashes, units and scope.
- `real_data_aggregation_checks.json`: verification and uploaded Colab comparison.
