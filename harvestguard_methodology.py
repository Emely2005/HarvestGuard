"""Public-facing methodology drawn from the recorded preprocessing and model run."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


SOURCES = [
    ("NISR Seasonal Agricultural Survey 2024", "RWA-NISR-SAS-2024-v01",
     "https://microdata.statistics.gov.rw/index.php/catalog/113", "ddi-documentation-english-113.pdf"),
    ("NISR Seasonal Agricultural Survey 2025", "RWA-NISR-SAS-2025-v01",
     "https://microdata.statistics.gov.rw/index.php/catalog/124", "ddi-documentation-english-124.pdf"),
]


def validation_summary(model_dir):
    """Report only the deployed model's recorded rate errors, with explicit units."""
    lock = json.loads((model_dir / "locked_choice_before_2025.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(model_dir / "external_validation_2025.csv")
    metadata = json.loads((model_dir / "model_v3_metadata.json").read_text(encoding="utf-8"))
    model_name = metadata.get("validation_model", "V3 locked candidate")
    selected = validation.loc[validation["scope"].eq("2025 all") & validation["model"].eq(model_name)]
    if len(selected) != 1:
        raise ValueError("The saved model must have one matching full-year validation row.")
    return pd.DataFrame([
        {"Evaluation": "2024 grouped cross-validation", "Profile error (percentage points)": lock["selection"]["mae_pp"],
         "District rate error (percentage points)": lock["selection"]["district_mae_pp"]},
        {"Evaluation": "2025 retrospective validation", "Profile error (percentage points)": selected.iloc[0]["mae_pp"],
         "District rate error (percentage points)": selected.iloc[0]["district_mae_pp"]},
    ])


def matched_importance(model_dir):
    directory = model_dir.parent / "model_v3_insights"
    evidence = json.loads((directory / "district_evidence.json").read_text(encoding="utf-8"))
    path = directory / "feature_importance_hurdle.csv"
    if (hashlib.sha256((model_dir / "model_v3_candidate.joblib").read_bytes()).hexdigest() != evidence["source_model_sha256"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != evidence["importance_csv_sha256"]):
        raise ValueError("Importance must match the displayed trained model.")
    return pd.read_csv(path)


def _download(path, label, key, mime):
    if path.is_file():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=key)


def render_metadata_methodology(metadata, model_dir):
    root = model_dir.parent
    data, methods, limitations, project = st.tabs(["Data sources", "Methodology", "Limitations", "Project & AI"])
    with data:
        st.markdown("### Data behind HarvestGuard")
        st.write("We combined NISR Seasonal Agricultural Survey data for 2024 and 2025 with district climate data. The analytical unit is one farmer–crop–season–year profile; the dashboard brings the results together by district, crop and season.")
        for title, reference, url, filename in SOURCES:
            st.markdown(f"**[{title}]({url})**  ·  `{reference}`")
            _download(root / filename, title.rsplit(" ", 1)[-1] + " metadata (PDF)", "metadata_" + filename, "application/pdf")
        st.markdown("**External data**")
        st.markdown("[CHIRPS rainfall](https://www.chc.ucsb.edu/data/chirps) · [NASA POWER humidity and temperature](https://power.larc.nasa.gov/data-access-viewer/) · [Rwanda district boundaries](https://www.geoboundaries.org/countryDownloads.html)")
        st.write("CHIRPS rainfall was averaged within district boundaries. NASA POWER relative humidity and air temperature were interpolated to district centroids using inverse-distance weights. Daily climate was joined by district and estimated harvest date. The displayed model uses rainfall during the preceding thirty days and humidity during the preceding seven days; temperature and realised weather after harvest belong to preparation, and are excluded from this model’s inputs.")
        st.caption("Survey citation: National Institute of Statistics of Rwanda (NISR), Seasonal Agricultural Survey 2024 and Seasonal Agricultural Survey 2025, edited anonymised public-use data, version 0.1. References and questionnaires are available through the linked catalogues.")
        quantities = root / metadata.get("prediction_file", "model_v3_quantity_results/harvestguard_2025_v3_quantity_priorities.csv")
        if quantities.is_file():
            frame = pd.read_csv(quantities)
            st.caption(f"Displayed coverage: {int(frame.n_profiles.sum()):,} scored profiles · {frame.district.nunique()} districts · {frame.crop_name.nunique()} crops · {len(frame):,} district–crop–season estimates in 2025.")

    with methods:
        st.markdown("### From survey records to district priorities")
        st.markdown("**1. Harmonise and clean.** We decoded Stata value labels, standardised district/crop names and identifiers, converted quantities to numeric values, and joined production and agricultural-practice modules with checked join cardinality. Plot records were combined into farmer–crop–season–year profiles to avoid counting repeated crop-level outcomes more than once. Negative losses, conflicting repeated production/loss values, missing target components and losses exceeding production were excluded from valid regression targets; missing outcomes were not changed to zero.")
        st.markdown("**2. Construct the target and climate features.** The target combines reported storage, processing and packaging losses relative to the same profile’s crop production. Survey harvest-period categories supplied a date midpoint for climate windows, with a flagged season midpoint where the category required it. Rainfall, humidity and temperature were prepared for historical analysis; only the two preharvest climate features enter the displayed model.")
        st.latex(r"y_i = 100\,\frac{L_{\mathrm{storage},i}+L_{\mathrm{processing},i}+L_{\mathrm{packaging},i}}{Q_i}")
        st.markdown("**3. Fit a two-part XGBoost model.** Many profiles report no loss, while positive losses are unevenly distributed. An occurrence classifier estimates the probability of any loss; a Gamma regression model estimates the percentage conditional on loss. Their product gives the expected loss percentage. The selected model uses ten predictors: season, district, crop, farmer type, harvest month, plot count, crop plot area, crops per plot, preharvest rainfall and preharvest humidity. Production quantities and loss-component columns are excluded from predictive inputs.")
        st.latex(r"\widehat{y}_i=P(y_i>0\mid x_i)\;\mathbb{E}[y_i\mid y_i>0,x_i]")
        st.markdown("**4. Preprocess inside training partitions.** Categorical variables were one-hot encoded, with infrequent levels pooled and unseen categories handled at prediction time. Numeric inputs used training-partition median imputation and missingness indicators. Training weights reduced the dominance of larger district–crop–season groups. These fitting weights are distinct from survey expansion weights.")
        st.markdown("**5. Tune, select and validate.** Development used 2024 observations and three-fold stratified group cross-validation: each district–crop–season group stayed within one fold. Early stopping used a separate grouped subset inside each training partition. Twelve Optuna search trials evaluated model settings and approaches; the selected configuration is the two-part model shown here. Its classifier and severity trees use depth three, with up to five hundred boosting rounds and early stopping. The chosen configuration uses no probability-calibration adjustment or historical blend. After selection, 2025 supplied retrospective temporal validation.")
        try:
            table = validation_summary(model_dir)
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                name: st.column_config.NumberColumn(format="%.4f") for name in table.columns if name != "Evaluation"
            })
            st.caption("Mean absolute error is the average absolute difference between a predicted and observed loss percentage; smaller values mean closer estimates. A percentage point measures an absolute difference between percentages. The table reports the selected model only.")
        except (OSError, ValueError, KeyError):
            st.info("The recorded validation summary is unavailable in this copy of the application.")
        st.latex(r"\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\widehat{y}_i|")
        st.markdown("**6. Convert percentages into loss volumes.** We multiply each profile’s expected loss percentage by its own production in kilograms, then sum those losses by district, crop and season. The district percentage is production-weighted. District priority follows total estimated kilograms lost.")
        st.latex(r"\widehat{L}_i=\frac{\widehat{y}_i}{100}Q_i,\qquad \widehat{L}_d=\sum_{i\in d}\widehat{L}_i,\qquad \widehat{y}_d=100\frac{\widehat{L}_d}{\sum_{i\in d}Q_i}")
        st.markdown("**7. Interpret district evidence.** We recomputed numeric-input comparisons with the unchanged model using 2024 crop–season median reference values. Production cohorts, harvest timing, peer comparisons, other crop losses and seasonal patterns supply additional evidence for the district explanation. The AI selects supported findings and actions and translates them into plain language; numerical values and evidence references are checked before display.")
        st.markdown("### Predictive importance")
        try:
            importance = matched_importance(model_dir).sort_values("combined_importance")
            label = "label" if "label" in importance else "feature"
            chart = px.bar(importance, x="combined_importance", y=label, orientation="h", color_discrete_sequence=["#326342"],
                           labels={"combined_importance": "Combined predictive importance", label: ""})
            chart.update_layout(height=420, margin=dict(l=0, r=10, t=10, b=10), paper_bgcolor="#fffdf7", plot_bgcolor="#fffdf7")
            st.plotly_chart(chart, width="stretch", key="predictive_importance")
            st.caption("Scores sum XGBoost split gain back to each original feature, normalise within the occurrence and severity components, then average the two components equally. They describe how the fitted model uses its predictors.")
        except (OSError, ValueError, KeyError):
            st.info("A model-matched importance chart is unavailable in this copy of the application.")
        _download(root / "source_preprocessing/build_postharvest_model_ready_corrected.py", "Download preprocessing code", "download_cleaning_code", "text/x-python")
        _download(root / "METHODOLOGY.md", "Download methodology", "download_methodology", "text/markdown")

    with limitations:
        st.markdown("### Interpretation and scope")
        st.markdown("- Quantities cover the sampled profiles and supported crops. Survey expansion weights were not supplied, so these are not whole-district or national loss totals.\n- The displayed estimates describe 2025 historical observations. Validation is retrospective; the application is not a live weather or crop forecast.\n- The recorded accuracy metrics evaluate loss percentages. The subsequent kilogram rankings have no separately reported ranking-accuracy estimate.\n- Season C has no reported positive-loss cases in the validation export, so discrimination between positive losses was not assessed for that season.\n- Predictive importance and one-input reference comparisons describe model associations. They do not identify causes, add up to a causal explanation, or measure an intervention’s effect.\n- The target combines storage, processing and packaging. It does not assign the predicted loss to an individual stage. Weather does not measure grain moisture, storage conditions or available infrastructure.\n- Detailed cohort summaries are suppressed for small groups. Estimates are point estimates; profile counts are not confidence intervals.\n- Recommended actions are planning priorities supported by the available evidence. Costs, treatment effects and avoided-loss benefits have not been measured.")

    with project:
        st.markdown("### Protecting harvests, supporting district decisions")
        st.write("HarvestGuard helps district agricultural teams, cooperatives and public planners identify where estimated handling-loss volumes are concentrated, when services are needed, and which evidence supports a response. Its focus on postharvest storage, processing and packaging aligns with the agricultural productivity and value-addition priorities of Rwanda’s NST2 agenda.")
        st.markdown("[NISR 2026 Hackathon: Agricultural Productivity track and submission requirements](https://statistics.gov.rw/about/hackathon/2026-hackathon-competition)")
        st.write("Practical workflow: select a season and crop, compare districts, open a district’s insights, then download its ranking and explanation for a planning discussion. A field pilot can record services delivered, quantities handled and measured losses to assess real-world impact.")
        st.markdown("#### AI assistance disclosure")
        st.write("The project idea, problem framing and decisions belong to the HarvestGuard team. OpenAI Codex was used as a development assistant for implementation, debugging, documentation and testing. Groq is used in the application to draft explanations from selected aggregate evidence. The team reviews the work and remains responsible for the submission, interpretations and source attribution.")
        st.caption("The AI receives aggregate district evidence and relevant project-method context. Individual farmer records, private settings and API keys are excluded from its evidence packet. Saved verified explanations are reused; model findings remain available when online generation is unavailable.")
