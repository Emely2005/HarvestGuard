# Insights evidence run — 22 October 2026

This document records the original evidence-export and verification run. The current application uses Groq for its automatic explanations and interactive **Ask HarvestGuard** questions; see `AI_SETUP.md`. The saved statistical model and aggregate evidence from this run are unchanged.

Generated with `harvestguard_insights_export.py` using the supplied production-bearing CSV and existing V3 candidate. No training or tuning was performed.

- Profiles: **50,475** eligible 2025 records.
- Output groups: **944** district/crop/season combinations.
- Production unit: user-confirmed **kg**; scope: **sample totals**.
- Recomputed group quantities agree with the saved quantity results.
- Original model SHA256 remains `6969c3c60a10571fd5e7644173f01568f9f496d64000beacb0fa0b7fee71eb64`.
- Global importance: crop, crop plot area and district lead; rainfall is fourth and humidity fifth. Normalization is documented in `district_evidence.json`.
- Local numeric sensitivities use the complete calibrated model and 2024 crop-season references. They are model comparisons, not causal effects or estimated losses avoided.

For Nyagatare maize, Season A: **64,545.97 kg**, rank **1/30**; largest-production **20.2%** of sampled profiles contain **98.3%** of estimated loss; the January harvest cohort contains **60.4%**. The selected actions prioritize high-volume batches and schedule support around this harvest cohort. This district does not satisfy the moisture-service selection rule.

The application retains the header and KPIs, promotes evidence-backed findings and actions, and groups scope, sources, methods, privacy and validation in collapsed metadata panels. Recommendations must cite their matching district evidence. No crop-only equipment checklist is used in the quantity view.

Verification: **98 automated tests passed**, with live Gemini generation, persistent verified-answer reuse, district filtering, metadata/privacy and desktop/mobile browser checks. Browser artifacts are stored locally under `.runtime/insights_redesign_*`. Numerical and citation checks also reject unsupported statistical-significance wording throughout the generated explanation.

Only aggregate selection evidence is sent to the approved Google Gemini connection. Farmer records and credentials are excluded. Detailed group summaries follow the small-group suppression rules recorded in the export metadata.
