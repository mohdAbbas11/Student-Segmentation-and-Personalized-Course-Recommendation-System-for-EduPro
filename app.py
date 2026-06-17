import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_all_data
from feature_engineering import build_learner_profiles, CLUSTER_FEATURES
from clustering import run_clustering, get_cluster_centers, SEGMENT_META
from recommender import build_recommendations, cluster_top_courses

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro · Learner Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1320 50%, #0a0f1e 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1628 0%, #0a1020 100%);
    border-right: 1px solid rgba(108,99,255,0.2);
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(0,210,255,0.08));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #00D2FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* Segment badge */
.seg-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Course card */
.course-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.course-card:hover {
    border-color: rgba(108,99,255,0.5);
    transform: translateX(4px);
}
.course-title { font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: 0.2rem; }
.course-meta  { font-size: 0.78rem; color: rgba(255,255,255,0.5); }
.course-reason { font-size: 0.76rem; color: #00D2FF; margin-top: 0.3rem; font-style: italic; }

/* Section header */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    border-left: 4px solid #6C63FF;
    padding-left: 0.8rem;
    margin: 1.4rem 0 1rem 0;
}

/* Divider */
hr { border-color: rgba(108,99,255,0.15) !important; }

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }

/* Plotly chart bg transparent */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Load & process data ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def get_processed_data():
    users, courses, transactions, teachers = load_all_data()
    profiles = build_learner_profiles(users, courses, transactions)
    labeled, pca2, pca3, elbow, optimal_k = run_clustering(profiles)
    return users, courses, transactions, teachers, profiles, labeled, pca2, pca3, elbow, optimal_k

with st.spinner("🚀 Initializing EduPro Intelligence Engine…"):
    (users, courses, transactions, teachers,
     profiles, labeled, pca2, pca3, elbow, optimal_k) = get_processed_data()

# ── Helpers ───────────────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    c: SEGMENT_META.get(c, {}).get("color", "#6C63FF")
    for c in labeled["Cluster"].unique()
}

def seg_name(c: int) -> str:
    m = SEGMENT_META.get(c, {})
    return f"{m.get('emoji','●')} {m.get('name', f'Segment {c}')}"

def seg_color(c: int) -> str:
    return SEGMENT_META.get(c, {}).get("color", "#6C63FF")

def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert #RRGGBB hex string to a valid rgba() CSS color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def rating_stars(r):
    if pd.isna(r): return "–"
    full = int(r)
    return "★" * full + "☆" * (5 - full)

def plotly_dark_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="rgba(255,255,255,0.8)", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)")
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.5rem 0;'>
        <div style='font-size:2.8rem;'>🎓</div>
        <div style='font-size:1.3rem; font-weight:800; color:#fff; letter-spacing:0.04em;'>EduPro</div>
        <div style='font-size:0.75rem; color:rgba(108,99,255,0.9); font-weight:500;
                    background:rgba(108,99,255,0.12); border-radius:999px;
                    padding:0.2rem 0.8rem; display:inline-block; margin-top:0.3rem;'>
            Learner Intelligence
        </div>
    </div>
    <hr style='margin: 1rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Overview",
            "🔬  Cluster Explorer",
            "👤  Learner Profile",
            "🎓  Recommendations",
            "📊  Segment Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.75rem; color:rgba(255,255,255,0.35); text-align:center;'>
        <b style='color:rgba(255,255,255,0.6);'>{len(labeled):,}</b> learners ·
        <b style='color:rgba(255,255,255,0.6);'>{optimal_k}</b> segments ·
        <b style='color:rgba(255,255,255,0.6);'>{len(courses)}</b> courses
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <h1 style='font-size:2rem;font-weight:800;color:#fff;margin:0;'>
            EduPro Learner Intelligence Hub
        </h1>
        <p style='color:rgba(255,255,255,0.45);font-size:0.9rem;margin:0.3rem 0 0 0;'>
            Student segmentation · Behavioral analytics · Personalized recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, f"{len(labeled):,}", "Total Learners"),
        (k2, str(optimal_k), "Segments Found"),
        (k3, f"{len(courses)}", "Courses"),
        (k4, f"{len(transactions):,}", "Transactions"),
        (k5, f"${labeled['total_spending'].mean():,.0f}", "Avg. Lifetime Spend"),
    ]
    for col, val, lbl in kpis:
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Segment donut + Age distribution ────────────────────────
    c1, c2 = st.columns([1, 1.6])

    with c1:
        st.markdown("<div class='section-header'>Learner Segments</div>", unsafe_allow_html=True)
        seg_counts = labeled["Cluster"].value_counts().reset_index()
        seg_counts.columns = ["Cluster", "Count"]
        seg_counts["Name"] = seg_counts["Cluster"].map(seg_name)
        seg_counts["Color"] = seg_counts["Cluster"].map(seg_color)

        fig_donut = go.Figure(go.Pie(
            labels=seg_counts["Name"],
            values=seg_counts["Count"],
            hole=0.62,
            marker=dict(colors=seg_counts["Color"].tolist(),
                        line=dict(color="#0a0e1a", width=3)),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value} learners (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="rgba(255,255,255,0.8)"),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            height=310,
            annotations=[dict(text=f"<b>{len(labeled):,}</b><br><span style='font-size:11px'>Learners</span>",
                              x=0.5, y=0.5, font_size=18, showarrow=False,
                              font_color="white")],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Age Distribution by Segment</div>", unsafe_allow_html=True)
        age_df = labeled[["Age", "Cluster"]].copy().dropna()
        age_df["Segment"] = age_df["Cluster"].map(seg_name)
        age_df["Color"] = age_df["Cluster"].map(seg_color)

        fig_age = px.histogram(
            age_df, x="Age", color="Segment",
            barmode="overlay", nbins=30,
            color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
            opacity=0.75,
        )
        plotly_dark_layout(fig_age, 310)
        fig_age.update_traces(marker_line_width=0)
        st.plotly_chart(fig_age, use_container_width=True)

    # ── Row 2: Gender breakdown + Spending ────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("<div class='section-header'>Gender Breakdown per Segment</div>", unsafe_allow_html=True)
        if "Gender" in labeled.columns:
            gdf = labeled.groupby(["Cluster", "Gender"]).size().reset_index(name="n")
            gdf["Segment"] = gdf["Cluster"].map(seg_name)
            fig_g = px.bar(
                gdf, x="Segment", y="n", color="Gender",
                barmode="stack",
                color_discrete_sequence=["#6C63FF", "#FF6584", "#00D2FF", "#FBBF24"],
            )
            plotly_dark_layout(fig_g, 310)
            st.plotly_chart(fig_g, use_container_width=True)

    with c4:
        st.markdown("<div class='section-header'>Avg. Spend per Segment</div>", unsafe_allow_html=True)
        spend_df = labeled.groupby("Cluster")["avg_spending"].mean().reset_index()
        spend_df["Segment"] = spend_df["Cluster"].map(seg_name)
        spend_df["Color"] = spend_df["Cluster"].map(seg_color)
        fig_sp = px.bar(
            spend_df, x="Segment", y="avg_spending",
            color="Segment",
            color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
            text_auto=".0f",
        )
        plotly_dark_layout(fig_sp, 310)
        fig_sp.update_traces(marker_line_width=0, textfont_color="white")
        fig_sp.update_layout(showlegend=False, yaxis_title="Avg Spend ($)")
        st.plotly_chart(fig_sp, use_container_width=True)

    # ── Segment summary cards ─────────────────────────────────────────
    st.markdown("<div class='section-header'>Segment Profiles</div>", unsafe_allow_html=True)
    cols = st.columns(min(optimal_k, 4))
    for i, c in enumerate(sorted(labeled["Cluster"].unique())):
        meta = SEGMENT_META.get(c, {"name": f"Segment {c}", "emoji": "●",
                                    "color": "#6C63FF", "description": "", "strategy": ""})
        n = (labeled["Cluster"] == c).sum()
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class='course-card' style='border-color:{meta["color"]}44;'>
                <div style='font-size:1.8rem;'>{meta["emoji"]}</div>
                <div style='font-size:1rem;font-weight:700;color:{meta["color"]};
                            margin:0.3rem 0 0.2rem 0;'>{meta["name"]}</div>
                <div style='font-size:0.78rem;color:rgba(255,255,255,0.55);
                            margin-bottom:0.5rem;'>{n:,} learners</div>
                <div style='font-size:0.78rem;color:rgba(255,255,255,0.7);
                            line-height:1.5;'>{meta["description"]}</div>
                <div style='margin-top:0.6rem;font-size:0.75rem;
                            color:{meta["color"]};font-style:italic;'>
                    💡 {meta["strategy"]}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  Cluster Explorer":
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:800;color:#fff;margin:0 0 0.3rem 0;'>
        Cluster Explorer
    </h1>
    <p style='color:rgba(255,255,255,0.45);font-size:0.9rem;margin:0 0 1.5rem 0;'>
        Dimensionality reduction & segment visualization
    </p>
    """, unsafe_allow_html=True)

    # Silhouette + Elbow
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Elbow Curve</div>", unsafe_allow_html=True)
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elbow["K"], y=elbow["Inertia"],
            mode="lines+markers",
            line=dict(color="#6C63FF", width=2.5),
            marker=dict(size=7, color="#6C63FF",
                        line=dict(color="#fff", width=1.5)),
            name="Inertia",
        ))
        fig_elbow.add_vline(x=optimal_k, line_dash="dot",
                            line_color="#00D2FF", opacity=0.8)
        fig_elbow.add_annotation(x=optimal_k, y=elbow["Inertia"].max() * 0.9,
                                  text=f"Optimal K={optimal_k}",
                                  font=dict(color="#00D2FF", size=12),
                                  showarrow=False)
        plotly_dark_layout(fig_elbow, 300)
        fig_elbow.update_layout(xaxis_title="K (clusters)", yaxis_title="Inertia")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Silhouette Scores</div>", unsafe_allow_html=True)
        best_idx = elbow["Silhouette"].idxmax()
        colors_sil = ["#FF6584" if i == best_idx else "#6C63FF"
                      for i in range(len(elbow))]
        fig_sil = go.Figure(go.Bar(
            x=elbow["K"], y=elbow["Silhouette"],
            marker_color=colors_sil,
            text=elbow["Silhouette"].round(3),
            textposition="outside",
            textfont=dict(color="white", size=11),
        ))
        plotly_dark_layout(fig_sil, 300)
        fig_sil.update_layout(xaxis_title="K (clusters)",
                               yaxis_title="Silhouette Score",
                               showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 2D PCA scatter
    st.markdown("<div class='section-header'>2D PCA — Learner Clusters</div>", unsafe_allow_html=True)
    pca2_plot = pca2.copy().reset_index()
    pca2_plot.columns = [str(c) for c in pca2_plot.columns]
    pca2_plot["Segment"] = pca2_plot["Cluster"].map(seg_name)

    fig_scatter2 = px.scatter(
        pca2_plot, x="PC1", y="PC2", color="Segment",
        color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
        hover_data={"UserID": True, "PC1": False, "PC2": False},
        opacity=0.7,
    )
    fig_scatter2.update_traces(marker=dict(size=5, line=dict(width=0)))
    plotly_dark_layout(fig_scatter2, 420)
    st.plotly_chart(fig_scatter2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # 3D PCA scatter
    st.markdown("<div class='section-header'>3D PCA — Learner Clusters</div>", unsafe_allow_html=True)
    pca3_plot = pca3.copy().reset_index()
    pca3_plot.columns = [str(c) for c in pca3_plot.columns]
    pca3_plot["Segment"] = pca3_plot["Cluster"].map(seg_name)

    fig_scatter3 = px.scatter_3d(
        pca3_plot, x="PC1", y="PC2", z="PC3", color="Segment",
        color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
        opacity=0.65,
    )
    fig_scatter3.update_traces(marker=dict(size=3))
    fig_scatter3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.07)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.07)"),
        ),
        font=dict(family="Inter", color="rgba(255,255,255,0.8)"),
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_scatter3, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Radar chart — cluster means
    st.markdown("<div class='section-header'>Cluster Feature Radar</div>", unsafe_allow_html=True)
    centers = get_cluster_centers(labeled)
    radar_feats = ["total_courses", "diversity_score", "depth_index",
                   "avg_spending", "avg_course_rating", "enrollment_frequency"]
    radar_feats = [f for f in radar_feats if f in centers.columns]

    fig_radar = go.Figure()
    for c in centers.index:
        vals = centers.loc[c, radar_feats].tolist()
        # Normalize per feature 0-1
        max_vals = centers[radar_feats].max()
        norm_vals = (centers.loc[c, radar_feats] / max_vals.clip(lower=1e-9)).tolist()
        norm_vals += norm_vals[:1]
        theta = radar_feats + radar_feats[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_vals, theta=theta,
            fill="toself", name=seg_name(c),
            line=dict(color=seg_color(c), width=2),
            fillcolor=hex_to_rgba(seg_color(c), alpha=0.18),
            opacity=0.85,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.1)",
                            color="rgba(255,255,255,0.4)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             color="rgba(255,255,255,0.6)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="rgba(255,255,255,0.8)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=420,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LEARNER PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Learner Profile":
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:800;color:#fff;margin:0 0 0.3rem 0;'>
        Learner Profile Explorer
    </h1>
    <p style='color:rgba(255,255,255,0.45);font-size:0.9rem;margin:0 0 1.5rem 0;'>
        Deep-dive into individual learner behavior and segment assignment
    </p>
    """, unsafe_allow_html=True)

    all_user_ids = sorted(labeled.index.tolist())
    selected_uid = st.selectbox("🔍 Select a Learner", all_user_ids,
                                 format_func=lambda x: f"User {x}")

    if selected_uid:
        row = labeled.loc[selected_uid]
        cluster = int(row["Cluster"])
        meta = SEGMENT_META.get(cluster, {"name": f"Segment {cluster}", "emoji": "●",
                                           "color": "#6C63FF", "description": "", "strategy": ""})

        # User info from users table
        user_info = users[users["UserID"] == selected_uid]
        uname = user_info["UserName"].values[0] if not user_info.empty and "UserName" in user_info.columns else selected_uid
        age_val = int(row.get("Age", 0)) if pd.notna(row.get("Age", None)) else "–"
        gender_val = row.get("Gender", "–")

        # ── Profile header ────────────────────────────────────────────
        h1, h2 = st.columns([1, 2])
        with h1:
            st.markdown(f"""
            <div class='course-card' style='border-color:{meta["color"]}66;text-align:center;padding:1.6rem;'>
                <div style='font-size:3.5rem;'>{meta["emoji"]}</div>
                <div style='font-size:1.2rem;font-weight:700;color:#fff;margin:0.5rem 0 0.2rem 0;'>{uname}</div>
                <div style='font-size:0.8rem;color:rgba(255,255,255,0.5);'>User ID: {selected_uid}</div>
                <div style='font-size:0.8rem;color:rgba(255,255,255,0.5);'>Age: {age_val} · Gender: {gender_val}</div>
                <div style='margin-top:0.8rem;'>
                    <span class='seg-badge' style='background:{meta["color"]}22;color:{meta["color"]};
                          border:1px solid {meta["color"]}55;'>
                        {meta["emoji"]} {meta["name"]}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with h2:
            st.markdown(f"""
            <div class='course-card' style='border-color:{meta["color"]}44;'>
                <div style='font-size:0.95rem;font-weight:600;color:{meta["color"]};
                            margin-bottom:0.5rem;'>Segment Description</div>
                <div style='font-size:0.85rem;color:rgba(255,255,255,0.75);line-height:1.6;'>
                    {meta["description"]}
                </div>
                <div style='margin-top:0.8rem;font-size:0.82rem;color:{meta["color"]};'>
                    💡 Strategy: {meta["strategy"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature metrics
            fm1, fm2, fm3, fm4 = st.columns(4)
            metrics = [
                (fm1, "📚", "Courses", f"{int(row.get('total_courses',0))}"),
                (fm2, "💰", "Total Spend", f"${row.get('total_spending',0):.0f}"),
                (fm3, "🌐", "Categories", f"{int(row.get('unique_categories',0))}"),
                (fm4, "⭐", "Avg Rating", f"{row.get('avg_course_rating',0):.1f}"),
            ]
            for col, icon, lbl, val in metrics:
                col.metric(label=f"{icon} {lbl}", value=val)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Feature bar chart ─────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-header'>Feature Profile</div>", unsafe_allow_html=True)
            feats = [f for f in CLUSTER_FEATURES if f in labeled.columns and f != "Age"]
            feat_df = pd.DataFrame({
                "Feature": feats,
                "User": [row.get(f, 0) for f in feats],
                "Segment Avg": [labeled[labeled["Cluster"] == cluster][f].mean() for f in feats],
            })
            fig_feat = go.Figure()
            fig_feat.add_trace(go.Bar(name="This Learner", x=feat_df["Feature"],
                                       y=feat_df["User"], marker_color=meta["color"], opacity=0.9))
            fig_feat.add_trace(go.Bar(name="Segment Avg", x=feat_df["Feature"],
                                       y=feat_df["Segment Avg"], marker_color="rgba(255,255,255,0.2)"))
            fig_feat.update_layout(barmode="group")
            plotly_dark_layout(fig_feat, 320)
            fig_feat.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_feat, use_container_width=True)

        with c2:
            st.markdown("<div class='section-header'>Category Enrollment Breakdown</div>", unsafe_allow_html=True)
            user_tx = transactions[transactions["UserID"] == selected_uid].merge(
                courses[["CourseID", "CourseCategory"]], on="CourseID", how="left"
            )
            if not user_tx.empty and "CourseCategory" in user_tx.columns:
                cat_counts = user_tx["CourseCategory"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                fig_cat = px.pie(cat_counts, names="Category", values="Count",
                                  color_discrete_sequence=["#6C63FF","#00D2FF","#FF6584",
                                                           "#FBBF24","#34D399","#F472B6"])
                fig_cat.update_traces(textinfo="percent+label",
                                      marker=dict(line=dict(color="#0a0e1a", width=2)))
                plotly_dark_layout(fig_cat, 320)
                st.plotly_chart(fig_cat, use_container_width=True)

        # ── Enrollment timeline ───────────────────────────────────────
        st.markdown("<div class='section-header'>Enrollment Timeline</div>", unsafe_allow_html=True)
        user_tl = transactions[transactions["UserID"] == selected_uid].copy()
        if not user_tl.empty and pd.api.types.is_datetime64_any_dtype(user_tl["TransactionDate"]):
            user_tl["Month"] = user_tl["TransactionDate"].dt.to_period("M").dt.to_timestamp()
            tl_agg = user_tl.groupby("Month").size().reset_index(name="Enrollments")
            fig_tl = px.area(tl_agg, x="Month", y="Enrollments",
                              color_discrete_sequence=[meta["color"]])
            fig_tl.update_traces(line_width=2, fillcolor=hex_to_rgba(meta['color'], alpha=0.15))
            plotly_dark_layout(fig_tl, 280)
            st.plotly_chart(fig_tl, use_container_width=True)
        else:
            st.info("Timeline data not available for this learner.")

        # ── Enrolled courses table ────────────────────────────────────
        st.markdown("<div class='section-header'>Courses Enrolled</div>", unsafe_allow_html=True)
        enrolled = user_tx.merge(
            courses[["CourseID", "CourseName", "CourseCategory",
                     "CourseLevel", "CourseRating", "CoursePrice"]],
            on="CourseID", how="left"
        ).drop_duplicates("CourseID")
        if not enrolled.empty:
            show_cols = [c for c in ["CourseName","CourseCategory","CourseLevel",
                                      "CourseRating","Amount"] if c in enrolled.columns]
            st.dataframe(
                enrolled[show_cols].rename(columns={"Amount": "Paid ($)"}),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎓  Recommendations":
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:800;color:#fff;margin:0 0 0.3rem 0;'>
        Personalized Recommendations
    </h1>
    <p style='color:rgba(255,255,255,0.45);font-size:0.9rem;margin:0 0 1.5rem 0;'>
        Cluster-aware · Rating-weighted · Level & category matched
    </p>
    """, unsafe_allow_html=True)

    all_user_ids = sorted(labeled.index.tolist())
    selected_uid = st.selectbox("🔍 Select a Learner", all_user_ids,
                                 format_func=lambda x: f"User {x}")

    if selected_uid:
        row = labeled.loc[selected_uid]
        cluster = int(row["Cluster"])
        meta = SEGMENT_META.get(cluster, {})

        # Filters
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            all_levels = ["All"] + sorted(courses["CourseLevel"].dropna().unique().tolist())
            level_f = st.selectbox("Level", all_levels)
        with f2:
            all_cats = ["All"] + sorted(courses["CourseCategory"].dropna().unique().tolist())
            cat_f = st.selectbox("Category", all_cats)
        with f3:
            all_types = ["All"] + sorted(courses["CourseType"].dropna().unique().tolist()) \
                        if "CourseType" in courses.columns else ["All"]
            type_f = st.selectbox("Type", all_types)
        with f4:
            top_n = st.slider("Top N", min_value=3, max_value=20, value=10)

        recs = build_recommendations(
            selected_uid, labeled, courses, transactions,
            top_n=top_n,
            level_filter=level_f,
            category_filter=cat_f,
            type_filter=type_f,
        )

        # Learner segment badge
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.8rem;margin:1rem 0;'>
            <span style='font-size:0.85rem;color:rgba(255,255,255,0.5);'>Segment:</span>
            <span class='seg-badge' style='background:{meta.get("color","#6C63FF")}22;
                  color:{meta.get("color","#6C63FF")};
                  border:1px solid {meta.get("color","#6C63FF")}55;'>
                {meta.get("emoji","●")} {meta.get("name","Unknown")}
            </span>
            <span style='font-size:0.8rem;color:rgba(255,255,255,0.4);'>
                Preferred: <b style='color:rgba(255,255,255,0.7);'>
                {row.get("preferred_category","–")}</b>
                · Level: <b style='color:rgba(255,255,255,0.7);'>
                {row.get("preferred_level","–")}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        if recs.empty:
            st.warning("No recommendations match the current filters. Try relaxing them.")
        else:
            # Two-column course cards
            col_a, col_b = st.columns(2)
            for i, r in recs.iterrows():
                price = f"${r['CoursePrice']:.0f}" if pd.notna(r.get("CoursePrice")) and r.get("CoursePrice",0) > 0 else "Free"
                rating = rating_stars(r.get("CourseRating"))
                match = r.get("match_pct", 0)
                bar_w = int(match)
                col = col_a if i % 2 == 0 else col_b
                with col:
                    st.markdown(f"""
                    <div class='course-card'>
                        <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                            <div class='course-title'>{r.get("CourseName","Course")}</div>
                            <div style='font-size:0.75rem;font-weight:700;
                                        background:rgba(108,99,255,0.2);
                                        color:#6C63FF;border-radius:8px;
                                        padding:0.2rem 0.6rem;white-space:nowrap;'>
                                {match:.0f}% match
                            </div>
                        </div>
                        <div class='course-meta'>
                            {r.get("CourseCategory","–")} ·
                            {r.get("CourseLevel","–")} ·
                            {r.get("CourseType","–")} ·
                            {rating} · {price}
                        </div>
                        <div style='margin:0.5rem 0 0.3rem 0;background:rgba(255,255,255,0.06);
                                    border-radius:4px;height:4px;'>
                            <div style='width:{bar_w}%;height:4px;border-radius:4px;
                                        background:linear-gradient(90deg,#6C63FF,#00D2FF);'></div>
                        </div>
                        <div class='course-reason'>✦ {r.get("reason","")}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # Match score bar chart
            st.markdown("<div class='section-header'>Match Score Breakdown</div>",
                        unsafe_allow_html=True)
            fig_match = px.bar(
                recs, x="match_pct",
                y=recs.get("CourseName", recs.index.astype(str)) if "CourseName" in recs.columns else recs.index.astype(str),
                orientation="h",
                color="match_pct",
                color_continuous_scale=[[0,"#6C63FF"],[1,"#00D2FF"]],
                text="match_pct",
            )
            fig_match.update_traces(texttemplate="%{text:.0f}%", textposition="outside",
                                    textfont_color="white")
            plotly_dark_layout(fig_match, max(280, len(recs) * 36))
            fig_match.update_layout(yaxis={"categoryorder":"total ascending"},
                                     coloraxis_showscale=False,
                                     xaxis_title="Match Score (%)",
                                     yaxis_title="")
            st.plotly_chart(fig_match, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SEGMENT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Segment Analytics":
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:800;color:#fff;margin:0 0 0.3rem 0;'>
        Segment Analytics
    </h1>
    <p style='color:rgba(255,255,255,0.45);font-size:0.9rem;margin:0 0 1.5rem 0;'>
        Cross-segment behavioral comparison and deep-dive analytics
    </p>
    """, unsafe_allow_html=True)

    # ── Segment size comparison ────────────────────────────────────────
    st.markdown("<div class='section-header'>Segment Size & Composition</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        seg_sizes = labeled["Cluster"].value_counts().reset_index()
        seg_sizes.columns = ["Cluster", "Count"]
        seg_sizes["Name"] = seg_sizes["Cluster"].map(seg_name)
        seg_sizes["Color"] = seg_sizes["Cluster"].map(seg_color)
        fig_bar = px.bar(
            seg_sizes, x="Name", y="Count", color="Name",
            color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
            text="Count",
        )
        plotly_dark_layout(fig_bar, 300)
        fig_bar.update_traces(marker_line_width=0, textfont_color="white",
                               textposition="outside")
        fig_bar.update_layout(showlegend=False, xaxis_tickangle=-15,
                               xaxis_title="", yaxis_title="Learners")
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        # Category heatmap per segment
        tx_merged = transactions.merge(
            courses[["CourseID", "CourseCategory"]], on="CourseID", how="left"
        ).merge(
            labeled[["Cluster"]].reset_index(), on="UserID", how="left"
        )
        tx_merged["Segment"] = tx_merged["Cluster"].map(seg_name)
        heat_df = tx_merged.groupby(["Segment", "CourseCategory"]).size().reset_index(name="n")
        heat_pivot = heat_df.pivot(index="Segment", columns="CourseCategory", values="n").fillna(0)

        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale=[[0,"rgba(108,99,255,0.05)"],[1,"#6C63FF"]],
            aspect="auto",
            text_auto=True,
        )
        plotly_dark_layout(fig_heat, 300)
        fig_heat.update_layout(
            coloraxis_showscale=False,
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Spending box plots ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>Spending Behavior Distribution</div>", unsafe_allow_html=True)
    spend_box = labeled[["avg_spending", "total_spending", "Cluster"]].copy()
    spend_box["Segment"] = spend_box["Cluster"].map(seg_name)

    c3, c4 = st.columns(2)
    with c3:
        fig_box1 = px.box(
            spend_box, x="Segment", y="avg_spending", color="Segment",
            color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
            points=False,
        )
        plotly_dark_layout(fig_box1, 320)
        fig_box1.update_layout(showlegend=False, xaxis_tickangle=-15,
                                yaxis_title="Avg Spending ($)", xaxis_title="")
        st.plotly_chart(fig_box1, use_container_width=True)

    with c4:
        fig_box2 = px.box(
            spend_box, x="Segment", y="total_spending", color="Segment",
            color_discrete_map={seg_name(c): seg_color(c) for c in labeled["Cluster"].unique()},
            points=False,
        )
        plotly_dark_layout(fig_box2, 320)
        fig_box2.update_layout(showlegend=False, xaxis_tickangle=-15,
                                yaxis_title="Total Spending ($)", xaxis_title="")
        st.plotly_chart(fig_box2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Engagement metrics stacked bar ────────────────────────────────
    st.markdown("<div class='section-header'>Level Preference per Segment</div>", unsafe_allow_html=True)
    lvl_df = tx_merged.merge(
        courses[["CourseID", "CourseLevel"]], on="CourseID", how="left"
    )
    lvl_df["Segment"] = lvl_df["Cluster"].map(seg_name)
    lvl_counts = lvl_df.groupby(["Segment","CourseLevel"]).size().reset_index(name="n")
    fig_lvl = px.bar(
        lvl_counts, x="Segment", y="n", color="CourseLevel",
        barmode="stack",
        color_discrete_sequence=["#6C63FF","#00D2FF","#FF6584","#FBBF24","#34D399"],
        text_auto=False,
    )
    plotly_dark_layout(fig_lvl, 340)
    fig_lvl.update_layout(xaxis_tickangle=-15, xaxis_title="", yaxis_title="Enrollments")
    st.plotly_chart(fig_lvl, use_container_width=True)

    # ── Engagement feature comparison table ───────────────────────────
    st.markdown("<div class='section-header'>Segment Feature Summary Table</div>", unsafe_allow_html=True)
    centers = get_cluster_centers(labeled)
    centers.index = centers.index.map(seg_name)
    display_feats = [f for f in CLUSTER_FEATURES if f in centers.columns]
    styled = centers[display_feats].round(2).style.bar(
        subset=display_feats,
        color=["#2d1b69", "#6C63FF"],
        axis=0,
    ).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)

    # ── Top courses per cluster ────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Top 5 Courses per Segment</div>", unsafe_allow_html=True)
    seg_cols = st.columns(min(optimal_k, 3))
    for i, c in enumerate(sorted(labeled["Cluster"].unique())):
        top_c = cluster_top_courses(c, labeled, courses, transactions, top_n=5)
        meta = SEGMENT_META.get(c, {})
        with seg_cols[i % len(seg_cols)]:
            st.markdown(f"""
            <div style='font-size:0.9rem;font-weight:700;
                        color:{meta.get("color","#6C63FF")};margin-bottom:0.5rem;'>
                {meta.get("emoji","●")} {meta.get("name",f"Segment {c}")}
            </div>
            """, unsafe_allow_html=True)
            if not top_c.empty and "CourseName" in top_c.columns:
                for _, row in top_c.iterrows():
                    st.markdown(f"""
                    <div class='course-card' style='padding:0.6rem 0.8rem;margin-bottom:0.4rem;'>
                        <div style='font-size:0.8rem;font-weight:600;color:#fff;'>
                            {row.get("CourseName","–")}
                        </div>
                        <div style='font-size:0.72rem;color:rgba(255,255,255,0.45);'>
                            {row.get("CourseCategory","–")} · {row.get("enrollments",0)} enrollments
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
