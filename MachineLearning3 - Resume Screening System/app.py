"""
app.py
------
IT Talent Screening Command Center — an operational Streamlit dashboard
for the Future Interns Machine Learning Task 3 resume screening system.

Run with:  streamlit run app.py
"""
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from scoring import build_pair_features, composite_score  # noqa: E402

ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

# ----------------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="IT Talent Screening Command Center",
    page_icon="\u25C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0A0E14",
    "panel": "#121821",
    "panel_alt": "#181F2A",
    "border": "#232C39",
    "text": "#E6EBF2",
    "muted": "#8593A6",
    "amber": "#E8A33D",
    "teal": "#3ECF8E",
    "coral": "#E2574C",
    "blue": "#4F8FE8",
}

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: {PALETTE['bg']};
    --panel: {PALETTE['panel']};
    --panel-alt: {PALETTE['panel_alt']};
    --border: {PALETTE['border']};
    --text: {PALETTE['text']};
    --muted: {PALETTE['muted']};
    --amber: {PALETTE['amber']};
    --teal: {PALETTE['teal']};
    --coral: {PALETTE['coral']};
    --blue: {PALETTE['blue']};
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #101823 0%, var(--bg) 45%) fixed;
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}}

section[data-testid="stSidebar"] {{
    background: var(--panel);
    border-right: 1px solid var(--border);
}}
section[data-testid="stSidebar"] * {{ color: var(--text) !important; }}

h1, h2, h3, h4, .eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.02em;
}}

/* Top banner */
.command-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 22px 28px;
    background: linear-gradient(120deg, var(--panel) 0%, var(--panel-alt) 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--amber);
    border-radius: 10px;
    margin-bottom: 18px;
}}
.command-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}}
.command-sub {{
    color: var(--muted);
    font-size: 0.88rem;
    margin-top: 4px;
}}
.status-pill {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--teal);
    border: 1px solid var(--teal);
    border-radius: 20px;
    padding: 5px 14px;
    letter-spacing: 0.08em;
}}
.status-pill::before {{ content: "\\25CF  "; }}

/* Section eyebrow labels */
.eyebrow {{
    color: var(--amber);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

/* KPI cards */
.kpi-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
}}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 4px;
}}
.kpi-value.amber {{ color: var(--amber); }}
.kpi-value.teal {{ color: var(--teal); }}
.kpi-sub {{ color: var(--muted); font-size: 0.76rem; margin-top: 2px; }}

/* Job detail panel */
.job-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}}
.job-title {{ font-size: 1.15rem; font-weight: 600; color: var(--text); }}
.job-meta {{ color: var(--muted); font-size: 0.85rem; margin-top: 2px; }}

/* Skill chips */
.chip {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 11px;
    border-radius: 20px;
    margin: 3px 5px 3px 0;
    border: 1px solid var(--border);
}}
.chip.required {{ border-color: var(--blue); color: var(--blue); }}
.chip.matched {{ border-color: var(--teal); color: var(--teal); background: rgba(62,207,142,0.08); }}
.chip.missing {{ border-color: var(--coral); color: var(--coral); background: rgba(226,87,76,0.08); }}

.panel-box {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
}}

hr {{ border-color: var(--border); }}

/* Streamlit widget tweaks */
.stSelectbox label, .stSlider label, .stMultiSelect label {{ color: var(--muted) !important; font-size: 0.82rem; }}
div[data-testid="stMetricValue"] {{ color: var(--amber); }}
footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=PALETTE["panel"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], family="IBM Plex Sans"),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        colorway=[PALETTE["amber"], PALETTE["teal"], PALETTE["blue"], PALETTE["coral"]],
        margin=dict(l=10, r=10, t=40, b=10),
    )
)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading screening engine...")
def load_artifacts():
    jobs = pd.read_pickle(os.path.join(ARTIFACTS, "jobs_processed.pkl"))
    resumes = pd.read_pickle(os.path.join(ARTIFACTS, "resumes_processed.pkl"))
    with open(os.path.join(ARTIFACTS, "tfidf_vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    with open(os.path.join(ARTIFACTS, "fit_model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    with open(os.path.join(ARTIFACTS, "model_metrics.json")) as f:
        metrics = json.load(f)
    return jobs, resumes, vec, bundle, metrics


jobs, resumes, vectorizer, model_bundle, metrics = load_artifacts()
model = model_bundle["model"]
FEATURE_COLS = model_bundle["feature_cols"]


@st.cache_data(show_spinner="Screening candidates...")
def screen_all(job_id, weights_key):
    job_row = jobs[jobs["job_id"] == job_id].iloc[0]
    weights = dict(zip(["skill_coverage", "tfidf_similarity", "experience_bonus"], weights_key))
    rows = []
    for _, r in resumes.iterrows():
        feats = build_pair_features(r, job_row, vectorizer)
        X_row = pd.DataFrame([{k: feats[k] for k in FEATURE_COLS}])
        if X_row["required_years"].iloc[0] == -1:
            X_row["required_years"] = 0
        ml_proba = float(model.predict_proba(X_row)[0][1])
        rows.append({
            "candidate_id": r["candidate_id"], "Candidate": r["name"],
            "Background": r["target_role_family"], "Seniority": r["seniority"],
            "Experience (yrs)": r["years_experience"], "Education": r["education_level"],
            "Composite Score": composite_score(feats, weights),
            "ML Fit %": round(ml_proba * 100, 1),
            "Skill Coverage %": round(feats["skill_coverage"] * 100, 1),
            "Matched": feats["_matched_skills"], "Missing": feats["_missing_skills"],
            "N Matched": len(feats["_matched_skills"]), "N Missing": len(feats["_missing_skills"]),
            "resume_text": r["resume_text"],
        })
    return pd.DataFrame(rows).sort_values("Composite Score", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### \u25C8 Screening Console")
    st.caption("Select a real job posting to screen the candidate pool against.")

    role_families = sorted(jobs["role_family"].unique())
    sel_family = st.selectbox("Role family", role_families, index=role_families.index("IT Support")
                               if "IT Support" in role_families else 0)

    family_jobs = jobs[jobs["role_family"] == sel_family].copy()
    family_jobs["label"] = family_jobs["job_title"] + "  ·  " + family_jobs["location"]
    sel_label = st.selectbox("Job posting", family_jobs["label"].tolist())
    sel_job = family_jobs[family_jobs["label"] == sel_label].iloc[0]

    st.markdown("---")
    st.markdown("**Composite Score weights**")
    w_skill = st.slider("Skill coverage", 0.0, 1.0, 0.55, 0.05)
    w_sem = st.slider("Semantic similarity", 0.0, 1.0, 0.30, 0.05)
    w_exp = st.slider("Experience fit", 0.0, 1.0, 0.15, 0.05)
    total_w = max(w_skill + w_sem + w_exp, 1e-6)
    weights_key = (w_skill / total_w, w_sem / total_w, w_exp / total_w)

    top_n = st.slider("Show top N candidates", 5, min(30, len(resumes)), 10)

    st.markdown("---")
    with st.expander("\U0001F4CA Model performance"):
        best = metrics["best_model"]
        st.caption(f"Ranking model: **{best}**")
        m = metrics["results"][best]
        c1, c2 = st.columns(2)
        c1.metric("F1 score", f"{m['F1']*100:.1f}%")
        c2.metric("ROC-AUC", f"{m['ROC-AUC']*100:.1f}%")
        c1.metric("Precision", f"{m['Precision']*100:.1f}%")
        c2.metric("Recall", f"{m['Recall']*100:.1f}%")
        st.caption(f"Trained on {metrics['n_training_pairs']:,} resume-job pairs "
                   f"({metrics['positive_rate']*100:.1f}% positive rate)")

    st.markdown("---")
    st.caption("Candidate pool is a simulated dataset (133 profiles) built for this task; "
               "job postings are real Monster.com listings.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="command-header">
    <div>
        <p class="command-title">IT TALENT SCREENING COMMAND CENTER</p>
        <p class="command-sub">NLP-driven resume screening &amp; candidate ranking · Future Interns ML Task 3</p>
    </div>
    <div class="status-pill">ENGINE ONLINE</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Job detail panel
# ----------------------------------------------------------------------------
required_skills = sel_job["skills_list"]
req_years_display = "not specified" if pd.isna(sel_job.get("req_years", np.nan)) else f"{int(sel_job['req_years'])}+ yrs"

chips_html = "".join(f'<span class="chip required">{s}</span>' for s in required_skills)
st.markdown(f"""
<div class="job-panel">
    <p class="eyebrow">Target Job Posting &middot; {sel_job['role_family']}</p>
    <p class="job-title">{sel_job['job_title']}</p>
    <p class="job-meta">{sel_job['organization']} &middot; {sel_job['location']} &middot; Experience required: {req_years_display}</p>
    <div style="margin-top:12px;">{chips_html}</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Screen + rank
# ----------------------------------------------------------------------------
ranked = screen_all(sel_job["job_id"], weights_key)
top_candidates = ranked.head(top_n)

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Candidates Screened</div>
    <div class="kpi-value">{len(ranked)}</div><div class="kpi-sub">across all backgrounds</div></div>""",
                unsafe_allow_html=True)
with k2:
    top_row = ranked.iloc[0]
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Top Candidate</div>
    <div class="kpi-value amber">{top_row['Composite Score']:.1f}</div>
    <div class="kpi-sub">{top_row['Candidate']}</div></div>""", unsafe_allow_html=True)
with k3:
    avg_ml = ranked["ML Fit %"].mean()
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avg ML Fit Probability</div>
    <div class="kpi-value teal">{avg_ml:.1f}%</div><div class="kpi-sub">model-learned signal</div></div>""",
                unsafe_allow_html=True)
with k4:
    strong_pool = (ranked["Skill Coverage %"] >= 50).mean() * 100
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Pool Meeting &ge;50% Skill Match</div>
    <div class="kpi-value">{strong_pool:.0f}%</div><div class="kpi-sub">of screened candidates</div></div>""",
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Ranked table
# ----------------------------------------------------------------------------
st.markdown('<p class="eyebrow">Ranked Candidates</p>', unsafe_allow_html=True)
display_cols = ["Candidate", "Background", "Seniority", "Experience (yrs)", "Composite Score",
                 "ML Fit %", "Skill Coverage %", "N Matched", "N Missing"]
st.dataframe(
    top_candidates[display_cols],
    hide_index=True,
    width='stretch',
    column_config={
        "Composite Score": st.column_config.ProgressColumn("Composite Score", min_value=0, max_value=100, format="%.1f"),
        "ML Fit %": st.column_config.ProgressColumn("ML Fit %", min_value=0, max_value=100, format="%.1f%%"),
        "Skill Coverage %": st.column_config.ProgressColumn("Skill Coverage %", min_value=0, max_value=100, format="%.0f%%"),
    },
)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Two-column: skill gap chart + candidate drilldown
# ----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown('<p class="eyebrow">Skill Gap Across Candidate Pool</p>', unsafe_allow_html=True)
    all_missing = [s for missing_list in ranked["Missing"] for s in missing_list]
    if all_missing:
        gap_counts = pd.Series(all_missing).value_counts().sort_values(ascending=True).tail(10)
        fig_gap = go.Figure(go.Bar(
            x=gap_counts.values, y=gap_counts.index, orientation="h",
            marker_color=PALETTE["coral"],
        ))
        fig_gap.update_layout(template=PLOTLY_TEMPLATE, height=360,
                               xaxis_title="Candidates missing this skill", yaxis_title="")
        st.plotly_chart(fig_gap, width='stretch')
    else:
        st.info("No skill gaps detected across the screened pool for this posting.")

    st.markdown('<p class="eyebrow" style="margin-top:8px;">Candidate Backgrounds in Top Results</p>',
                unsafe_allow_html=True)
    bg_counts = top_candidates["Background"].value_counts()
    fig_bg = go.Figure(go.Pie(labels=bg_counts.index, values=bg_counts.values, hole=0.55,
                               marker=dict(colors=px.colors.sequential.Sunset)))
    fig_bg.update_layout(template=PLOTLY_TEMPLATE, height=300, showlegend=True,
                          legend=dict(font=dict(size=10)))
    st.plotly_chart(fig_bg, width='stretch')

with col_right:
    st.markdown('<p class="eyebrow">Candidate Drilldown</p>', unsafe_allow_html=True)
    cand_options = [f"#{i+1}  ·  {row['Candidate']}  ({row['Composite Score']:.1f})"
                    for i, row in top_candidates.reset_index(drop=True).iterrows()]
    sel_cand_label = st.selectbox("Select candidate", cand_options, label_visibility="collapsed")
    sel_idx = cand_options.index(sel_cand_label)
    cand = top_candidates.reset_index(drop=True).iloc[sel_idx]

    matched, missing = cand["Matched"], cand["Missing"]
    categories = matched + missing
    if categories:
        radar_values = [1] * len(matched) + [0.15] * len(missing)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values, theta=categories, fill="toself",
            fillcolor="rgba(62,207,142,0.25)", line=dict(color=PALETTE["teal"]),
            name="Skill match",
        ))
        fig_radar.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            polar=dict(bgcolor=PALETTE["panel"],
                       radialaxis=dict(visible=False, range=[0, 1]),
                       angularaxis=dict(color=PALETTE["text"], tickfont=dict(size=9))),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, width='stretch')

    chip_html = "".join(f'<span class="chip matched">\u2713 {s}</span>' for s in matched)
    chip_html += "".join(f'<span class="chip missing">\u2715 {s}</span>' for s in missing)
    st.markdown(f'<div class="panel-box">{chip_html or "No skill data available."}</div>',
                unsafe_allow_html=True)

    with st.expander("View resume text"):
        st.write(cand["resume_text"])

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built for Future Interns Machine Learning Task 3 · Job postings sourced from the real "
           "Monster.com jobs dataset · Candidate pool is simulated per the task's own data guidance.")

