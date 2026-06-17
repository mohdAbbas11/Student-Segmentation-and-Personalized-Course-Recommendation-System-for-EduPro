import pandas as pd
import numpy as np


def build_recommendations(
    user_id: str,
    labeled_profiles: pd.DataFrame,
    courses: pd.DataFrame,
    transactions: pd.DataFrame,
    top_n: int = 10,
    level_filter: str = "All",
    category_filter: str = "All",
    type_filter: str = "All",
) -> pd.DataFrame:
    """
    Generate personalized course recommendations for a given user.

    Strategy:
    1. Find the user's cluster
    2. Identify popular courses within that cluster (cluster-popularity score)
    3. Boost courses matching user's preferred category and level
    4. Penalize courses the user already enrolled in
    5. Apply optional filters and return top-N ranked courses
    """
    if user_id not in labeled_profiles.index:
        return pd.DataFrame()

    user_row = labeled_profiles.loc[user_id]
    user_cluster = int(user_row["Cluster"])
    preferred_cat = str(user_row.get("preferred_category", ""))
    preferred_lvl = str(user_row.get("preferred_level", ""))

    # Courses already taken
    taken = set(transactions[transactions["UserID"] == user_id]["CourseID"].astype(str))

    # All users in the same cluster
    cluster_users = labeled_profiles[labeled_profiles["Cluster"] == user_cluster].index.tolist()

    # Cluster transaction subset
    cluster_tx = transactions[transactions["UserID"].isin(cluster_users)].copy()
    cluster_tx["CourseID"] = cluster_tx["CourseID"].astype(str)

    # Course popularity within cluster (enrollment count)
    cluster_pop = (
        cluster_tx.groupby("CourseID")["UserID"]
        .count()
        .reset_index()
        .rename(columns={"UserID": "cluster_enrollments"})
    )

    # Merge with course metadata
    reco = courses.copy()
    reco["CourseID"] = reco["CourseID"].astype(str)
    reco = reco.merge(cluster_pop, on="CourseID", how="left")
    reco["cluster_enrollments"] = reco["cluster_enrollments"].fillna(0)

    # Remove already-taken courses
    reco = reco[~reco["CourseID"].isin(taken)]

    if reco.empty:
        return pd.DataFrame()

    # ── Scoring ────────────────────────────────────────────────────────
    max_enroll = reco["cluster_enrollments"].max() or 1
    max_rating = reco["CourseRating"].max() or 1

    reco["norm_popularity"] = reco["cluster_enrollments"] / max_enroll
    reco["norm_rating"] = reco["CourseRating"].fillna(0) / max_rating

    # Category boost (+0.3 if matches preferred)
    reco["cat_boost"] = reco["CourseCategory"].apply(
        lambda c: 0.3 if c == preferred_cat else 0.0
    )

    # Level boost (+0.2 if matches preferred)
    reco["lvl_boost"] = reco["CourseLevel"].apply(
        lambda l: 0.2 if l == preferred_lvl else 0.0
    )

    reco["score"] = (
        0.40 * reco["norm_popularity"]
        + 0.35 * reco["norm_rating"]
        + reco["cat_boost"]
        + reco["lvl_boost"]
    )

    # ── Filters ────────────────────────────────────────────────────────
    if level_filter != "All":
        reco = reco[reco["CourseLevel"] == level_filter]
    if category_filter != "All":
        reco = reco[reco["CourseCategory"] == category_filter]
    if type_filter != "All":
        reco = reco[reco["CourseType"] == type_filter]

    if reco.empty:
        return pd.DataFrame()

    top = reco.sort_values("score", ascending=False).head(top_n).copy()

    # Reason text
    def reason(row):
        parts = []
        if row["cat_boost"] > 0:
            parts.append(f"matches your fav category ({preferred_cat})")
        if row["lvl_boost"] > 0:
            parts.append(f"aligns with your level ({preferred_lvl})")
        if row["norm_popularity"] > 0.6:
            parts.append("very popular in your learner group")
        if row["norm_rating"] > 0.8:
            parts.append("highly rated")
        return " · ".join(parts) if parts else "popular among similar learners"

    top["reason"] = top.apply(reason, axis=1)
    top["match_pct"] = (top["score"] * 100).clip(upper=99).round(1)

    return top.reset_index(drop=True)


def cluster_top_courses(
    cluster_id: int,
    labeled_profiles: pd.DataFrame,
    courses: pd.DataFrame,
    transactions: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the top courses for a given cluster by enrollment count."""
    cluster_users = labeled_profiles[labeled_profiles["Cluster"] == cluster_id].index
    cluster_tx = transactions[transactions["UserID"].isin(cluster_users)]
    pop = (
        cluster_tx.groupby("CourseID")["UserID"]
        .count()
        .reset_index()
        .rename(columns={"UserID": "enrollments"})
    )
    pop["CourseID"] = pop["CourseID"].astype(str)
    result = pop.merge(courses, on="CourseID", how="left")
    return result.sort_values("enrollments", ascending=False).head(top_n)
