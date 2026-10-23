# HarvestGuard Rwanda: submission brief

**Problem.** Agricultural planners need to distinguish high loss percentages from large volumes of food lost, then connect district priorities to practical seasonal services. HarvestGuard focuses on storage, processing and packaging, supporting the postharvest and value-addition scope of the NISR Agricultural Productivity track and NST2 priorities.

**Solution.** Select a crop and season, compare estimated loss kg across districts, open a district interpretation, and download the ranking, explanation and evidence. The saved model estimates loss percentages; production-weighted conversion supplies volume priorities. District findings combine local model comparisons, climate context, loss concentration, harvest-cohort timing, other supported crops and seasonal patterns.

| Evaluation criterion | Demonstrated project evidence |
| --- | --- |
| Problem understanding and relevance | A specific handling-loss target, district planning workflow, and clear alignment with postharvest collection, storage, processing and value addition. |
| Data use and methodology | NISR 2024/2025 survey catalogues and supplied DDI PDFs; CHIRPS and NASA POWER climate; original preprocessing code and audit; grouped development validation, retrospective temporal evaluation, formulas and selected-model errors. |
| Technical innovation | A two-part XGBoost model, per-profile kg conversion before district aggregation, model-matched numeric reference comparisons, aggregate evidence retrieval and validated AI interpretation. |
| Usability and design | A district map, linked filters, concise priorities, understandable findings, downloadable outputs and a single metadata/methodology section with the predictive-importance chart. |
| Tangible impact potential | District teams and cooperatives can use a common loss-volume view to plan seasonal handling services. A monitored pilot can record service delivery, quantities handled and observed losses. Avoided-loss benefits have not yet been measured. |

The verified coverage is 50,475 scored 2025 profiles, 30 districts, 15 crops and 944 district–crop–season estimates. These are sample quantities, not whole-district totals. Model errors are reported in percentage points and are not a percentage accuracy claim or a predicted judging score. Evaluation and competition outcomes depend on the judges; no score is guaranteed.

## Suggested demonstration

1. Open Maize, Season A. Explain why the map ranks kg rather than percentages.
2. Open Nyagatare to show its loss volume and the evidence for its selected programme priorities.
3. Switch to Nyarugenge and show how a high loss percentage can coexist with a much smaller loss volume. Compare the district's crop portfolio and seasonal pattern.
4. Open Metadata and Methodology to show NISR documentation, the recorded selected-model errors, formulas, climate provenance and predictive importance.
5. Download a district explanation and ranking for a planning discussion. Describe a pilot measurement workflow without claiming unmeasured benefits.

## AI assistance disclosure

The project idea, problem framing and decisions belong to the HarvestGuard team. OpenAI Codex was used as a development assistant for implementation, debugging, documentation and testing. Google Gemini drafts plain-language explanations from selected aggregate evidence in the application. The team reviews the work, acknowledges sources and remains responsible for the submission and interpretations.

The [official competition page](https://statistics.gov.rw/about/hackathon/2026-hackathon-competition) requires a public GitHub repository, a deployed application link and documentation. It gives a submission deadline of **30 October 2026, 11:59 PM**. Confirm the team meets participation and originality requirements when submitting.
