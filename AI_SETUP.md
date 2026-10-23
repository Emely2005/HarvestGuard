# Enable HarvestGuard's generative AI demo

The application has **Map** and **Insights** screens. The working key and model are configured in private server settings. The dashboard contains no key-entry, connection-setup or model-selection controls.

The dashboard uses the downloaded V3 candidate, verified kg-loss results and real aggregate evidence in `model_v3_insights/`. Live verification on October 22 generated, validated and saved the revised intervention explanation for Nyagatare maize, Season A. Candidate identity and evidence are part of the cache; V2 wording is not reused. Replacing the configured key clears older session overrides.

## Get a free-tier key

1. Open [Google AI Studio's API keys page](https://aistudio.google.com/apikey) and sign in.
2. Create a key in your project. Google's [API key guide](https://ai.google.dev/gemini-api/docs/api-key) explains project creation/import and key access.
3. Keep the project on the **Free tier** if you want to avoid charges. The default model is `gemini-3.1-flash-lite`; Google's [pricing page](https://ai.google.dev/gemini-api/docs/pricing), checked October 21, 2026, lists free-tier input and output for that model. Limits and availability depend on your account and may change. The app cannot inspect your billing tier.
4. Keep the key in `.streamlit/secrets.toml`, an environment variable or the host's secret settings. Do not put the real key in source code or this documentation. The dashboard never asks visitors for a key.

Rwanda is listed in Google's [supported regions](https://ai.google.dev/gemini-api/docs/available-regions). The free service has quotas, requires internet, and may use prompts and responses to improve Google's products. Use public or non-sensitive evidence for the demo. There is no automatic paid fallback.

## Start the demo

In your project folder and Python environment:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run harvestguard_app.py
```

Then:

1. Select **Season A** and **Maize** in the sidebar.
2. Select **Nyagatare** in the sidebar.
3. Open **Insights**, or click **See insights →** beside the map.
4. The page displays **Where to intervene first**, **AI insights** and **Recommended actions**. Change a selection to display its matching findings.
5. Open **AI explanation & evidence** for provenance, methods and generation details. Model comparisons remain under **Metadata and Methodology**. Downloads preserve the evidence trail.

Map does not call the API. Insights displays verified findings immediately and generates wording in the background or reuses a validated answer from `.runtime/ai_cache/`. Navigation stays responsive. Background jobs are shared and serialized. Saved explanations survive browser refreshes and app restarts while that folder remains available. Changed evidence, prompt, model or private-key scope requires a matching entry. Source links appear with sourced practices, in metadata and in downloads.

When the V3 bundle supplies production and loss quantities, briefs and AI packets rank districts by summed estimated kg lost, retain the percentage as context, and include `sample_total` or `survey_expanded` scope. Numeric placeholders include `loss_kg`, `loss_tonnes`, `production_kg` and `median_loss_kg`. The AI must use the ranking basis in the evidence packet. Without quantities, the view compares historical rates and cannot establish priorities by kg. The advisor prompt version changed so cached rate-based explanations cannot be reused as quantity-based explanations.

Connection/server failures get one bounded retry. Google retry timing and shared cooldowns are respected. One automatic recovery cycle can follow a short cooldown; daily quota and access errors require the underlying condition to change. Technical status and **Retry AI** are under **AI explanation & evidence → AI & privacy**. Model findings and selected actions remain visible while new wording is pending. A failed evidence check receives one correction attempt; unverified text is withheld.

Check your current project limits in [AI Studio](https://aistudio.google.com/rate-limit). [Google applies rate limits per project](https://ai.google.dev/gemini-api/docs/rate-limits), so replacing a key in the same project does not reset quota. A working key does not guarantee uninterrupted free API availability. No paid fallback or billing change is made by this app.

For persistent configuration, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace the placeholder privately:

```toml
GEMINI_API_KEY = "YOUR_PRIVATE_KEY"
GEMINI_MODEL = "gemini-3.1-flash-lite"
```

Alternatively set the `GEMINI_API_KEY` environment variable before starting Streamlit. For hosting, use the host's secrets settings. `.gitignore` excludes the local secrets file. Never commit the real key. With no key, maps, comparisons, published context and rule-based evidence briefs still work.

## What the AI reads

| Input | Available in this demo | What it supports |
|---|---|---|
| Saved loss percentage and profile count | Yes | Explain the expected rate, units and sample support. |
| Other districts' saved kg losses for the same crop and season | Yes | Explain the kg ranking, comparison median and first intervention locations. |
| Model-wide feature importance | Extracted from this saved candidate | Explain global predictive patterns with the documented per-component normalization. |
| Selected actions | Evidence-rule selector | Recommend only actions backed by matching district findings; sourced agronomic practice links are supplied when applicable. |
| Published district context | Nyagatare, maize, Season A 2025 | Explain planning relevance using a linked NISR source. |
| Field observations | Supported by the evidence-packet API; no entry form in the simplified dashboard | Future integrations can supply dated, attributed observations without claiming a model input. |
| Local numeric sensitivities | Recomputed with the saved calibrated model | Compare the prediction with a 2024 crop-season reference; not causal effects, SHAP values or estimated intervention benefits. |
| Aggregate weather and production cohorts | Real production-bearing input | Explain ambient humidity/rainfall, concentration of loss volume and harvest-cohort service windows. |
| Verified storage capacity and grain moisture | Not supplied | Do not invent facility shortages, storage conditions or grain moisture measurements. |

Use **Download evidence & metadata** under **AI explanation & evidence → Methods** to inspect the packet. It excludes farmer-level records, model weights and credentials.

## A persuasive explanation that the evidence supports

This is an example of the intended explanation, not a stored Gemini response:

Current quantity-based example from the real export:

“For maize in Season A 2025, Nyagatare ranks first by estimated loss volume: 64,545.97 kg, representing 0.79% of covered production.

“The largest-production 20.2% of sampled profiles account for 98.3% of estimated loss. Prioritize high-volume batches in the processing, storage and packaging programme. Align service capacity with the January harvest cohort, which accounts for 60.4% of estimated loss.”

The NISR context is in [knowledge_base/district_context.json](knowledge_base/district_context.json), sourced to [SAS 2025 Season A, Table 14](https://statistics.gov.rw/sites/default/files/documents/2025-05/SAS%202025A_Final%20%20report.pdf#page=37). Its production estimate is external planning context. It has not been joined to compatible production weights for the model's profile averages, so the app does not multiply it into a district loss-total claim.

## Add actual local conditions

The simplified dashboard has no observation-entry form. The evidence-packet API still supports dated, attributed observations for a future integration. Historical observations must match the selected year; a later follow-up cannot explain a historical prediction. User observations are not independently verified.

For reusable public evidence, add records to `knowledge_base/district_context.json` with unique ID, district, crop, season, year, factual text, source title, HTTPS source URL and scope. The app matches all four selection fields. Guidance about another district or year must not silently become evidence for this selection.

## Reproduce local model evidence

Run `python harvestguard_insights_export.py` with the saved model's dependencies. It verifies the production input and unchanged model against the quantity manifest, reproduces the saved quantities and exports aggregate evidence to `model_v3_insights/`. No training or tuning occurs. Read the export metadata for numeric reference substitutions, importance normalization, units and small-group suppression. Changing model inputs is a model comparison, not evidence that an intervention will change real losses.

### Optional alternative attribution export

The AI does not learn the internals of XGBoost merely by receiving its JSON weights. Export local contributions from the same feature rows and preprocessing used for the saved predictions.

The optional file is `model_v3_candidate/local_explanations.json`. Its structure is an object with `schema_version: 1` and a `records` array. Each record needs:

- `district`, `crop`, `season`, `year`, and `expected_loss_pct` matching the selected saved prediction.
- `method`: how contributions were computed.
- `aggregation`: how profile-level explanations were aggregated to the district estimate, including any weights.
- `drivers`: records with `feature`, finite numeric `contribution`, `unit`, and `component` (`occurrence`, `severity`, or `expected_loss`).

Use the actual component's units. Occurrence-model SHAP values in log-odds are not percentage points of expected loss. Do not add two components' attributions and call that an explanation of their product without a justified method. The app verifies identity and basic structure, not the mathematical validity of your attribution method.

No example driver values are loaded as real data. The dashboard prefers the verified `model_v3_insights/` evidence. If it is absent, only a genuinely matching alternative export can support local attribution claims; global importance alone cannot.

## Troubleshooting

| Symptom | Action |
|---|---|
| Model findings appear before AI wording | Inspect generation details under AI explanation & evidence → AI & privacy. Saved verified answers are reused; temporary failures expire automatically. |
| Authentication/access error | Verify the key and project/API permissions in AI Studio. |
| “Your project has been denied access” | Google is blocking the key's project. Check project status and contact Google support. A new key in the same blocked project may not resolve it. Use an eligible project with confirmed access. |
| Model unavailable | Choose an available text model supporting structured output; check its free-tier status before changing. |
| Quota/rate-limit message | Check AI Studio limits. Short limits get cooldown/recovery; exhausted daily quota cannot be fixed by immediate retries or another key in the same project. |
| The generated draft could not be verified | The app attempts one correction automatically. If unsuccessful, it shows the evidence brief and withholds the generated draft. Unknown numbers, sources and interventions are not silently accepted. |
| New selection is still preparing AI wording | Findings and data-backed actions appear immediately; the matching AI explanation updates automatically. No Generate button is required. |

The REST implementation follows Google's [generation reference](https://ai.google.dev/api/generate-content) and [structured-output documentation](https://ai.google.dev/gemini-api/docs/generate-content/structured-output). Structural and citation checks reduce common errors but do not prove every generated sentence is supported; review the briefing before presenting it.
