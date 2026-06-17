import pandas as pd
import numpy as np


def build_learner_profiles(users: pd.DataFrame,
                           courses: pd.DataFrame,
                           transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to UserID-level learner profiles.
    Returns a DataFrame indexed by UserID with engineered features.
    """

    # Merge transactions with course metadata
    tx = transactions.merge(
        courses[["CourseID", "CourseCategory", "CourseLevel",
                 "CourseType", "CourseRating", "CoursePrice"]],
        on="CourseID", how="left"
    )

    # ── Engagement features ────────────────────────────────────────────
    total_courses = tx.groupby("UserID")["CourseID"].count().rename("total_courses")

    unique_cats = tx.groupby("UserID")["CourseCategory"].nunique().rename("unique_categories")

    # Enrollment frequency: transactions per active month
    if pd.api.types.is_datetime64_any_dtype(tx["TransactionDate"]):
        tx["YearMonth"] = tx["TransactionDate"].dt.to_period("M")
        active_months = tx.groupby("UserID")["YearMonth"].nunique().rename("active_months")
    else:
        active_months = pd.Series(1, index=tx["UserID"].unique(), name="active_months")

    # ── Preference features ────────────────────────────────────────────
    preferred_category = (
        tx.groupby("UserID")["CourseCategory"]
        .agg(lambda x: x.value_counts().idxmax() if len(x) > 0 else "Unknown")
        .rename("preferred_category")
    )

    preferred_level = (
        tx.groupby("UserID")["CourseLevel"]
        .agg(lambda x: x.value_counts().idxmax() if len(x) > 0 else "Unknown")
        .rename("preferred_level")
    )

    avg_course_rating = tx.groupby("UserID")["CourseRating"].mean().rename("avg_course_rating")

    # ── Behavioral / Financial features ───────────────────────────────
    avg_spending = tx.groupby("UserID")["Amount"].mean().rename("avg_spending")
    total_spending = tx.groupby("UserID")["Amount"].sum().rename("total_spending")

    # Diversity score: unique categories / total courses  (0–1)
    diversity_score = (unique_cats / total_courses.clip(lower=1)).rename("diversity_score")

    # Depth index: fraction of Advanced-level courses
    advanced_count = (
        tx[tx["CourseLevel"].str.lower().str.contains("advanced", na=False)]
        .groupby("UserID")["CourseID"].count()
    )
    depth_index = (advanced_count / total_courses.clip(lower=1)).fillna(0).rename("depth_index")

    # % free courses
    free_mask = tx["Amount"] == 0
    free_count = tx[free_mask].groupby("UserID")["CourseID"].count()
    pct_free = (free_count / total_courses.clip(lower=1)).fillna(0).rename("pct_free")

    # ── Assemble profile table ─────────────────────────────────────────
    profile = pd.concat(
        [total_courses, unique_cats, active_months,
         preferred_category, preferred_level,
         avg_course_rating, avg_spending, total_spending,
         diversity_score, depth_index, pct_free],
        axis=1
    ).fillna(0)

    # Merge with user demographics
    users_indexed = users.set_index("UserID")
    profile = profile.join(users_indexed[["Age", "Gender"]], how="left")

    # Compute enrollment_frequency (avoid div-by-zero)
    profile["enrollment_frequency"] = (
        profile["total_courses"] / profile["active_months"].clip(lower=1)
    )

    profile = profile.fillna({"Age": profile["Age"].median(),
                               "Gender": "Unknown",
                               "avg_course_rating": 0,
                               "avg_spending": 0})

    profile.index.name = "UserID"
    return profile


# Numeric columns used for clustering
CLUSTER_FEATURES = [
    "total_courses",
    "unique_categories",
    "enrollment_frequency",
    "avg_course_rating",
    "avg_spending",
    "total_spending",
    "diversity_score",
    "depth_index",
    "pct_free",
    "Age",
]
