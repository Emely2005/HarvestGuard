# HarvestGuard AI configuration

The application uses **Groq** for both automatic district explanations and **Ask HarvestGuard**. Groq is a different provider from xAI Grok. A Groq key must be used with Groq's API; changing a Google key alone is insufficient.

## Private server settings

Set these environment variables or use the ignored `.streamlit/secrets.toml` locally. For the hosted application, use **Streamlit Community Cloud > App settings > Secrets**:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Replace the placeholder privately. The application has no API-key entry field, and real credentials must never be committed or placed in a public file. `HARVESTGUARD_AI_ENABLED=0` disables online AI requests for isolated review. No Google credential is used by the current interface, and there is no provider fallback.

The model must support Groq strict structured outputs. The tested default uses JSON-schema responses and low reasoning effort. Requests go only to `https://api.groq.com/openai/v1/chat/completions`, using the key in an Authorization header. There is no new SDK dependency. See [Groq API reference](https://console.groq.com/docs/api-reference) and [structured output support](https://console.groq.com/docs/structured-outputs).

## Ask HarvestGuard

Click the floating chat icon at the bottom right of any page, or open the **Ask HarvestGuard** page for a larger view. Both share the same conversation for the selected filters. Choose a district, crop and season, then type a question or choose an example. An officer can ask, for example, "Why does Nyagatare rank first for Sorghum?" The assistant checks the actual saved rank rather than accepting the question's premise.

Explicit district, crop and season names override the default context for that question. Follow-ups keep the last single selection within that chat. Changing sidebar filters starts a separate conversation; **Clear chat** removes that thread. Histories are bounded to ten exchanges per thread and eight filter combinations per browser session. Conversations are not stored in the shared explanation cache or logged by the app.

Each answer links to the saved aggregate quantities, matched insight evidence or checked-in methodology it cites. Numerical values are inserted from trusted tokens tied to the cited facts. Unsupported numbers, links, source IDs, statistical-significance claims, asserted causes and invented infrastructure conditions are rejected, with at most one correction attempt. A user can inspect sources, download an answer and choose a follow-up question.

The assistant reads only the aggregate data already validated by the application and three explicit public documents: `METHODOLOGY.md`, `MODEL_REVIEW.md` and `PROJECT_AUDIT.md`. It cannot read arbitrary paths, open private settings, execute tools, browse or change data. No farmer-level CSV is retrieved. Questions and a relevant subset of aggregate evidence are sent to Groq. Avoid putting personal information into questions.

These checks establish structural and numerical consistency; they do not prove every natural-language inference. The model must acknowledge unavailable evidence, retrospective coverage and the difference between model sensitivity and causation. The original data-preparation gaps remain visible in the project audit.

## Automatic district explanations

Insights still generates evidence-backed narratives and selects supplied planning actions. The Groq adapter reuses the original identity, numerical, citation and action checks. Verified narratives are cached under the evidence, prompt, model and private configuration. Keys never enter caches. The previous Gemini transport implementation is retained only for compatibility and regression coverage inside the shared grounding module; the current interface calls the Groq adapter exclusively.

## Limits and recovery

Groq enforces account-specific request and token limits. Requests retrieve a bounded subset of evidence, use bounded output limits and retry only brief transient failures. Longer rate-limit waits are shown as a cooldown; the app does not silently change provider or model. Ask about fewer districts or one crop/season if a selection is too large. A provider outage leaves Map, deterministic Insights, methodology and downloads available. Open questions require a working server credential and available provider quota.

The release builder explicitly includes the question modules and tests and scans public text for Groq and other credential patterns. `.streamlit/secrets.toml`, `.runtime/`, private CSVs and individual profile predictions are excluded from GitHub.
