"""Question interface with selection-aware, session-local conversation history."""
import json
import time

import streamlit as st

from harvestguard_ai import AIError
from harvestguard_ai_view import _setting
from harvestguard_chat import (
    evidence_sources, generate_answer, render_paragraph, retrieve_evidence, thread_id, validate_question,
)
from harvestguard_groq import DEFAULT_MODEL


def _show_answer(entry, index, widget_prefix=""):
    if "error" in entry:
        st.info(entry["error"])
        return
    answer, evidence = entry["answer"], entry["evidence"]
    for paragraph in answer["paragraphs"]:
        st.markdown(render_paragraph(paragraph, evidence))
    with st.expander("Sources for this answer"):
        for source in evidence_sources(answer, evidence):
            st.markdown(f"**[{source['source_title']}]({source['source_url']})**")
            st.caption(render_paragraph({"text": source["text"]}, evidence))
    st.caption("Answered with Groq · Checked against saved project evidence")
    markdown = "# Ask HarvestGuard\n\n" + entry["question"] + "\n\n"
    markdown += "\n\n".join(render_paragraph(paragraph, evidence) for paragraph in answer["paragraphs"])
    markdown += "\n\n## Sources\n\n" + "\n".join(
        f"- [{source['source_title']}]({source['source_url']})" for source in evidence_sources(answer, evidence))
    st.download_button("Download answer", markdown, file_name="harvestguard_answer.md", mime="text/markdown", key=f"{widget_prefix}chat_download_{index}")


def render_project_assistant(data, metadata, app_dir, model_dir, selection, *, widget_prefix="", compact=False):
    st.caption(f"Current context: {selection['district']} · {selection['crop']} · Season {selection['season']} · 2025")
    if not compact:
        st.write("Ask about a ranking, a model finding, the data or the methods. You can name another district, crop or season in your question.")
    api_key = _setting("GROQ_API_KEY").strip() if _setting("HARVESTGUARD_AI_ENABLED", "1").strip() != "0" else ""
    model = _setting("GROQ_MODEL", DEFAULT_MODEL).strip()
    identity = thread_id(selection, model, api_key)
    threads = st.session_state.setdefault("_project_chat_threads", {})
    thread = threads.setdefault(identity, {"messages": [], "selection": dict(selection), "retry_at": 0})
    while len(threads) > 8:
        old = next(key for key in threads if key != identity)
        del threads[old]
    left, right = st.columns([3, 2] if compact else [4, 1])
    with right:
        if st.button("Clear chat", key=f"{widget_prefix}clear_project_chat", disabled=not thread["messages"]):
            threads[identity] = {"messages": [], "selection": dict(selection), "retry_at": 0}
            st.rerun()
    pending = None
    with st.container(height=300 if compact else "content", border=False,
                      key=f"{widget_prefix}chat_messages", autoscroll=compact):
        if not thread["messages"]:
            examples = [f"Why is {selection['district']} ranked here for {selection['crop']}?",
                        "What is the difference between loss percentage and kilograms lost?",
                        "How were the datasets cleaned and the model evaluated?"]
            for index, question in enumerate(examples):
                if st.button(question, key=f"{widget_prefix}chat_example_{index}", width="stretch"):
                    pending = question
        for index, entry in enumerate(thread["messages"]):
            with st.chat_message(entry["role"]):
                if entry["role"] == "user":
                    st.write(entry["text"])
                else:
                    _show_answer(entry, index, widget_prefix)
        if thread["messages"] and "answer" in thread["messages"][-1]:
            for index, question in enumerate(thread["messages"][-1]["answer"]["follow_up_questions"]):
                if st.button(question, key=f"{widget_prefix}chat_followup_{len(thread['messages'])}_{index}"):
                    pending = question
    if not api_key:
        st.info("Live answers are temporarily unavailable. You can still explore the verified model findings on Insights.")
    question = st.chat_input("Ask HarvestGuard about this project…", key=f"{widget_prefix}project_question", max_chars=1000,
                             disabled=not api_key)
    pending = question or pending
    if not pending:
        return
    if not api_key:
        st.info("Live answers are temporarily unavailable. Please try again later or open Insights for the saved findings.")
        return
    try:
        pending = validate_question(pending)
    except AIError as error:
        st.info(str(error))
        return
    if time.time() < thread["retry_at"]:
        st.info("The AI service is cooling down. Please try your question again shortly.")
        return
    history = []
    for entry in thread["messages"][-8:]:
        if entry["role"] == "user":
            history.append({"role": "user", "text": entry["text"]})
        elif "answer" in entry:
            history.append({"role": "assistant", "text": " ".join(render_paragraph(p, entry["evidence"]) for p in entry["answer"]["paragraphs"])[:2400]})
    thread["messages"].append({"role": "user", "text": pending})
    with st.chat_message("user"):
        st.write(pending)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading the project evidence and checking the answer…"):
                evidence = retrieve_evidence(pending, thread["selection"], data, metadata, app_dir, model_dir)
                answer = generate_answer(evidence, api_key, model=model, history=history)
            entry = {"role": "assistant", "question": pending, "answer": answer, "evidence": evidence}
            thread["messages"].append(entry)
            if len(evidence["selections"]) == 1:
                current = evidence["selections"][0]
                thread["selection"] = {key: current[key] for key in ("district", "crop", "season")}
        except AIError as error:
            thread["messages"].append({"role": "assistant", "error": str(error)})
            thread["retry_at"] = time.time() + (error.retry_after or 0)
        except (OSError, ValueError, KeyError, TypeError):
            thread["messages"].append({"role": "assistant", "error": "The matching project evidence could not be verified. Please choose another selection or try a more specific question."})
    # Bound each browser session; questions are not written to shared disk caches.
    thread["messages"] = thread["messages"][-20:]
    st.rerun()


def render_floating_assistant(data, metadata, app_dir, model_dir, selection):
    """Keep project chat within reach without leaving the current dashboard page."""
    with st.container(key="hg_assistant_launcher"):
        panel = st.popover("Ask HarvestGuard", icon=":material/chat:",
                           help="Ask HarvestGuard about the current selection",
                           key="hg_assistant_open", on_change="rerun")
        if panel.open:
            with panel, st.container(key="hg_assistant_panel"):
                st.markdown("### Ask HarvestGuard")
                render_project_assistant(data, metadata, app_dir, model_dir, selection,
                                         widget_prefix="floating_", compact=True)
