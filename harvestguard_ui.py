"""Shared visual language for the two HarvestGuard screens."""

from html import escape

import streamlit as st


def apply_design():
    st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background:#f6f5ed; color:#203b2e; }
[data-testid="stHeader"] { background:rgba(246,245,237,.94); }
.block-container { max-width:1440px; padding-top:3.8rem; padding-bottom:2rem; }
[data-testid="stSidebar"] { background:#e9eddf; border-right:1px solid #d7decc; }
[data-testid="stSidebar"] .block-container { padding-top:2rem; }
h1,h2,h3,h4 { color:#173f2b; letter-spacing:-.025em; }
h3 { font-size:1.3rem !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#566650; }
[data-testid="stCaptionContainer"] a { color:#345e30; }
[data-testid="stMetric"] {
    background:#fffdf7; border:1px solid #dce1d2; border-top:3px solid #81985b;
    padding:.85rem 1.1rem; border-radius:14px; min-height:110px;
}
[data-testid="stMetricValue"] { color:#214b32; font-size:1.8rem; }
[data-testid="stMetricLabel"] { color:#566650; }
[data-testid="stExpander"] { background:#fffdf7; border-radius:12px; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
.hg-brand { padding:.4rem 0 1.3rem; }
.hg-brand strong { display:block; font-size:1.5rem; color:#173f2b; letter-spacing:-.04em; }
.hg-brand small { color:#5c6b4d; letter-spacing:.12em; font-size:.64rem; }
.hg-hero {
    position:relative; overflow:hidden; color:#fffdf1; border-radius:22px;
    padding:1.6rem 2rem; margin-bottom:1.3rem;
    background:radial-gradient(ellipse at 90% 130%,#738746 0,transparent 48%),#183e2c;
}
.hg-hero::after {
    content:""; position:absolute; right:4%; top:-70px; width:220px; height:320px;
    border:1px solid #b8cb863b; border-radius:100% 0 100% 0; transform:rotate(25deg);
    pointer-events:none;
}
.hg-eyebrow { color:#d5dda8; text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; }
.hg-hero h1 { color:#fffdf1; font-size:2.2rem; line-height:1.1; margin:.4rem 0; padding:0; }
.hg-hero p { color:#e1e8d0; margin:.5rem 0 0; max-width:620px; font-size:.95rem; }
.hg-pill { display:inline-block; border:1px solid #b9c88b6b; border-radius:99px; padding:.23rem .7rem;
    color:#e5eec9; margin-top:.9rem; font-size:.7rem; }
.hg-card { background:#fffdf7; border:1px solid #dce1d2; border-radius:16px;
    padding:1.1rem 1.25rem; margin-bottom:.8rem; }
.hg-card.green { background:#e8efdf; border-color:#d4dfc5; }
.hg-card.gold { background:#f7edce; border-color:#e9d8a5; }
.hg-card .hg-kicker { color:#647747; text-transform:uppercase; letter-spacing:.12em; font-size:.65rem; font-weight:800; }
.hg-card h4 { font-size:1.08rem; margin:.3rem 0 .55rem; padding:0; color:#21442d; }
.hg-card p { color:#344b39; font-size:.92rem; line-height:1.55; margin:0; }
.hg-card .hg-check { border-top:1px solid #d6decb; padding-top:.65rem; margin-top:.8rem; font-size:.8rem; }
.hg-card a { color:#345e30; font-size:.76rem; }
.hg-card .hg-source { margin-top:.65rem; }
.hg-card .hg-evidence { border-top:1px solid #dce1d2; padding-top:.7rem; margin-top:.8rem; font-size:.82rem; color:#52634d; }
.hg-priorities { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.75rem; margin:.3rem 0 .65rem; }
.hg-priority { background:#fffdf7; border:1px solid #dce1d2; border-radius:14px; padding:1rem 1.15rem; }
.hg-priority.first { background:#193f2d; border-color:#193f2d; }
.hg-priority small { display:block; color:#65764e; font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.hg-priority b { display:block; color:#234730; font-size:1.13rem; margin:.4rem 0 .2rem; }
.hg-priority span { color:#855a2b; font-size:1rem; font-variant-numeric:tabular-nums; }
.hg-priority.first small { color:#c1d29a; }
.hg-priority.first b,.hg-priority.first span { color:#fffdf1; }
.hg-priority-note { color:#58664e; font-size:.81rem; margin:.3rem 0 1.3rem; }
.hg-rank { display:flex; align-items:center; gap:.7rem; margin:.85rem 0 .35rem; }
.hg-rank b { flex:1; color:#26442f; font-size:.9rem; }
.hg-rank span { color:#7b4a23; font-weight:700; font-size:.9rem; }
.hg-rank small { background:#e8eddc; color:#4b6239; border-radius:50%; width:25px; height:25px;
    display:inline-flex; align-items:center; justify-content:center; font-size:.75rem; }
.hg-track { height:5px; border-radius:10px; background:#e9eddf; margin-left:2.2rem; }
.hg-track div { height:100%; border-radius:10px; background:#c98642; }
.hg-legend { display:flex; flex-wrap:wrap; gap:1rem; font-size:.72rem; color:#566650; margin:.3rem 0; }
.hg-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:.3rem; }
.hg-footer { font-size:.72rem; color:#6b775d; margin-top:1.3rem; border-top:1px solid #dce1d2; padding-top:.8rem; }
.st-key-hg_assistant_launcher {
    position:fixed; right:1.5rem; bottom:1.5rem; width:auto !important; z-index:1000;
}
.st-key-hg_assistant_launcher [data-testid="stPopoverButton"] {
    width:60px; height:60px; padding:0; border-radius:50%; border:1px solid #50734a;
    background:#193f2d; color:#fffdf1; box-shadow:0 5px 22px #173f2b40;
}
.st-key-hg_assistant_launcher [data-testid="stPopoverButton"]:hover {
    background:#2b5738; color:#fffdf1; border-color:#81985b;
}
.st-key-hg_assistant_launcher [data-testid="stPopoverButton"]:focus-visible {
    outline:3px solid #c98642; outline-offset:4px;
}
.st-key-hg_assistant_launcher [data-testid="stPopoverButton"] [data-testid="stMarkdownContainer"] {
    position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%);
}
.st-key-hg_assistant_launcher [data-testid="stPopoverButton"] div[aria-hidden="true"] { display:none; }
[data-testid="stPopoverBody"]:has(.st-key-hg_assistant_panel) {
    width:430px; max-width:calc(100vw - 2rem); max-height:calc(100dvh - 110px);
    overflow-y:auto; border:1px solid #d7decc; border-radius:18px; background:#fffdf7;
    box-shadow:0 10px 45px #173f2b26;
}
.st-key-floating_chat_messages {
    height:clamp(100px, calc(100dvh - 370px), 320px) !important;
}
[data-testid="stBottomBlockContainer"] { padding-right:6.5rem; }
@media(max-width:800px) {
    .hg-priorities { grid-template-columns:1fr; }
    .block-container { padding-top:3.7rem; }
    .hg-hero { padding:1.4rem; }
    .hg-hero h1 { font-size:1.8rem; }
    .hg-card { padding:1rem; }
    [data-testid="stMetric"] { min-height:88px; padding:.65rem 1rem; }
    .st-key-hg_assistant_launcher { right:1rem; bottom:1rem; }
}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle, tag):
    st.markdown(
        f'<div class="hg-hero"><div class="hg-eyebrow">HarvestGuard Rwanda · 🌾</div>'
        f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p>'
        f'<span class="hg-pill">{escape(tag)}</span></div>', unsafe_allow_html=True,
    )


def card(kicker, title, text, *, tone="", check="", source=None, evidence=""):
    """Escape model and data text before placing it inside presentation HTML."""
    detail = f'<p class="hg-check"><strong>Check first:</strong> {escape(check)}</p>' if check else ""
    if evidence:
        detail += f'<p class="hg-evidence"><strong>Based on our data</strong><br>{escape(evidence)}</p>'
    link = ""
    if source and source[1].startswith("https://"):
        link = f'<div class="hg-source"><a href="{escape(source[1], quote=True)}" target="_blank" rel="noopener noreferrer">{escape(source[0])} ↗</a></div>'
    tone = tone if tone in {"green", "gold"} else ""
    st.markdown(
        f'<div class="hg-card {tone}"><div class="hg-kicker">{escape(kicker)}</div>'
        f'<h4>{escape(title)}</h4><p>{escape(text)}</p>{detail}{link}</div>',
        unsafe_allow_html=True,
    )


def priority_strip(leaders, share, *, selected):
    rows = []
    for index, row in enumerate(leaders, start=1):
        rank = row.get("rank", index)
        tied_first = rank == 1 and sum(item.get("rank") == 1 for item in leaders) > 1
        label = "Equal first" if tied_first else "Start here" if index == 1 else f"Priority {rank}"
        district = escape(row["district"])
        rows.append(f'<div class="hg-priority {"first" if rank == 1 else ""}"><small>{label}</small>'
                    f'<b>{district}</b><span>{row["loss_kg"]:,.2f} kg estimated loss</span></div>')
    st.markdown('<div class="hg-priorities">' + ''.join(rows) + '</div>'
                f'<p class="hg-priority-note">These districts account for <strong>{share:.1f}%</strong> of estimated loss in this crop and season.</p>',
                unsafe_allow_html=True)


def ranking_cards(view):
    quantity = "predicted_loss_kg" in view
    column = "predicted_loss_kg" if quantity else "predicted_loss_pct"
    peak = float(view[column].max())
    rows = []
    for _, row in view.loc[view.display_rank <= 5].iterrows():
        value = float(row[column])
        label = f"{value:,.2f} kg" if quantity else f"{value:.2f}%"
        width = 100 * value / peak if peak > 0 else 0
        rows.append(
            f'<div class="hg-rank"><small>{int(row["display_rank"])}</small>'
            f'<b>{escape(str(row["district"]))}</b><span>{label}</span></div>'
            f'<div class="hg-track"><div style="width:{width:.2f}%"></div></div>'
        )
    st.markdown(''.join(rows), unsafe_allow_html=True)
