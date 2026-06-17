import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from feature_engineering import CLUSTER_FEATURES


# ── Segment metadata ──────────────────────────────────────────────────────────
SEGMENT_META = {
    0: {
        "name": "Casual Explorers",
        "emoji": "🌱",
        "color": "#00D2FF",
        "description": (
            "Learners who browse broadly across many categories "
            "but don't go deep. Mostly free courses, low total spend."
        ),
        "strategy": "Introduce curated learning paths to build consistency.",
    },
    1: {
        "name": "Specialist Learners",
        "emoji": "🎯",
        "color": "#6C63FF",
        "description": (
            "Focused learners who stick to one or two categories "
            "and pursue courses at multiple levels within them."
        ),
        "strategy": "Recommend advanced and certification tracks in their domain.",
    },
    2: {
        "name": "Career Climbers",
        "emoji": "💼",
        "color": "#FF6584",
        "description": (
            "High-spending, advanced-level learners who invest in "
            "career-boosting certifications and professional courses."
        ),
        "strategy": "Surface premium bundles, mentorship, and job-placement courses.",
    },
    3: {
        "name": "Beginner Discoverers",
        "emoji": "🔰",
        "color": "#FBBF24",
        "description": (
            "New-to-platform learners with low engagement, beginner-only "
            "enrollments, and mostly free course preferences."
        ),
        "strategy": "Onboard with free starter packs and gentle progression prompts.",
    },
    4: {
        "name": "Power Learners",
        "emoji": "⚡",
        "color": "#34D399",
        "description": (
            "High-frequency, high-diversity learners who enroll across "
            "many categories and have very high total spending."
        ),
        "strategy": "Offer subscription plans and cross-category bundles.",
    },
}


@st.cache_data(show_spinner="🔬 Running clustering pipeline…", ttl=3600)
def run_clustering(profile_df: pd.DataFrame, k_range=(2, 8)):
    """
    Run K-Means clustering on learner profiles.
    Returns: (labeled_df, pca_df, elbow_data, sil_scores, optimal_k)
    """
    df = profile_df.copy()

    # Use only available numeric features
    features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X = df[features].fillna(0)

    # ── Scale ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow + Silhouette sweep ───────────────────────────────────────
    inertias, sil_scores, k_values = [], [], []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
        k_values.append(k)

    # Optimal K = highest silhouette
    optimal_k = k_values[int(np.argmax(sil_scores))]

    # ── Final model ───────────────────────────────────────────────────
    final_km = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    df["Cluster"] = final_km.fit_predict(X_scaled)
    df["Cluster"] = df["Cluster"].astype(int)

    # ── PCA for 2-D visualization ─────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        coords, columns=["PC1", "PC2"], index=df.index
    )
    pca_df["Cluster"] = df["Cluster"]
    pca_df["ClusterName"] = pca_df["Cluster"].map(
        lambda c: f"{SEGMENT_META.get(c, {}).get('emoji','●')} "
                  f"{SEGMENT_META.get(c, {}).get('name', f'Segment {c}')}"
    )

    # ── PCA 3-D ───────────────────────────────────────────────────────
    pca3 = PCA(n_components=3, random_state=42)
    coords3 = pca3.fit_transform(X_scaled)
    pca3_df = pd.DataFrame(
        coords3, columns=["PC1", "PC2", "PC3"], index=df.index
    )
    pca3_df["Cluster"] = df["Cluster"]
    pca3_df["ClusterName"] = pca3_df["Cluster"].map(
        lambda c: f"{SEGMENT_META.get(c, {}).get('emoji','●')} "
                  f"{SEGMENT_META.get(c, {}).get('name', f'Segment {c}')}"
    )

    elbow_data = pd.DataFrame(
        {"K": k_values, "Inertia": inertias, "Silhouette": sil_scores}
    )

    return df, pca_df, pca3_df, elbow_data, optimal_k


def get_cluster_centers(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """Return mean feature values per cluster."""
    features = [f for f in CLUSTER_FEATURES if f in labeled_df.columns]
    return labeled_df.groupby("Cluster")[features].mean()
