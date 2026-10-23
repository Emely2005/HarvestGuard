"""Streamlit views for auditable decision support and saved model evidence."""

import json
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from harvestguard_decisions import build_brief, brief_to_markdown, load_knowledge_base


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.info(f"Evidence file unavailable: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return pd.DataFrame()


def render_decision_support(view: pd.DataFrame, metadata: dict, model_dir: Path, kb_path: Path, district: str):
    st.markdown("### District decision brief")
    st.caption(
        "Evidence brief · Open the AI advisor tab for generative explanations · "
        "Pilot intervention evidence covers maize and tomato"
    )
    row = view.loc[view["district"] == district].iloc[0]
    try:
        kb = load_knowledge_base(kb_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.warning(f"Intervention evidence is unavailable. No decision brief was generated: {error}")
        return

    diagnostics = _read_optional_csv(model_dir / "external_validation_by_season_2025.csv")
    positive_cases = None
    if {"season", "positive_cases"}.issubset(diagnostics.columns):
        selected = diagnostics.loc[diagnostics["season"].astype(str) == str(row["season"])]
        if not selected.empty and pd.notna(selected.iloc[0]["positive_cases"]):
            positive_cases = int(selected.iloc[0]["positive_cases"])
    try:
        brief = build_brief(
            row.to_dict(), metadata, kb, district_count=len(view),
            season_positive_cases=positive_cases,
        )
    except (ValueError, KeyError, TypeError) as error:
        st.warning(f"The saved prediction cannot be used in a decision brief: {error}")
        return

    st.write(brief["interpretation"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected loss rate", f"{brief['expected_loss_pct']:.2f}%")
    c2.metric("Profiles in this estimate", f"{brief['n_profiles']:,}")
    c3.metric("Sample support", brief["sample_support"])
    st.caption(
        f"Observed loss in the 2025 validation data: {brief['observed_loss_pct']:.2f}%. "
        "Sample support reflects record count, not calibrated confidence."
    )
    if brief["sample_support"] in {"Limited", "Low"}:
        st.warning("This estimate has few profiles. Verify it locally before using it to prioritise a pilot.")
    if positive_cases == 0:
        st.warning(
            f"Season {brief['season']} has zero reported positive-loss cases in the 2025 validation export. "
            "Ranking performance for districts with losses could not be assessed."
        )

    st.markdown("#### What explains this estimate?")
    st.write(brief["local_explanation"])
    st.write(brief["weather_context"])
    st.caption(
        "The Model evidence tab shows importance across the whole model. "
        "That does not establish a local cause, its direction, or an intervention effect."
    )
    st.markdown("#### Next decision")
    st.info(brief["next_step"])
    if brief["interventions"]:
        st.markdown("#### Evidence-based options to assess")
        st.caption(
            "Options match the crop. The sequence follows prerequisites for assessment; "
            "it does not rank benefits or establish a district-specific need."
        )
        for index, option in enumerate(brief["interventions"], start=1):
            with st.expander(f"{index}. {option['title']} · {option['stage']}", expanded=index == 1):
                st.write(option["action"])
                st.markdown(f"**Consider when:** {option['when_to_consider']}")
                st.markdown(f"**Check first:** {option['check_first']}")
                st.markdown(f"**Evidence:** {option['evidence_summary']}")
                source = option["source"]
                st.markdown(f"[Read source: {source['title']}]({source['url']})")
                st.caption(
                    f"{source['evidence_type']}. {source['scope']} "
                    f"Source checked {source['reviewed_on']}; Rwanda agronomic review pending."
                )

    with st.expander("Evidence gaps and limits of this brief"):
        for limitation in brief["limitations"]:
            st.write(f"• {limitation}")
        st.write(brief["hurdle_components"])
    stem = f"harvestguard_{brief['district']}_{brief['crop']}_{brief['season']}_2025".replace(" ", "_")
    left, right = st.columns(2)
    left.download_button(
        "Download decision brief", brief_to_markdown(brief).encode("utf-8"),
        file_name=f"{stem}.md", mime="text/markdown", key="download_brief",
    )
    right.download_button(
        "Download structured evidence", json.dumps(brief, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"{stem}.json", mime="application/json", key="download_evidence",
    )
    return brief


def render_model_evidence(metadata: dict, readiness: pd.DataFrame, model_dir: Path):
    """Render the single-model public methodology; keep the call signature stable."""
    from harvestguard_methodology import render_metadata_methodology
    render_metadata_methodology(metadata, model_dir)
