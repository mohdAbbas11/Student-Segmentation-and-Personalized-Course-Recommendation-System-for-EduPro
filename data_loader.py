import pandas as pd
import streamlit as st

SHEET_ID = "1Ts1Yjt9oiTdOeCeuJPbExNz7N3vmd0Ln"

def _csv_url(sheet_name: str) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
        f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )

@st.cache_data(show_spinner="📥 Loading EduPro dataset…", ttl=3600)
def load_all_data():
    """Load all four sheets from the public Google Sheet and return cleaned DataFrames."""
    try:
        users = pd.read_csv(_csv_url("Users"))
        courses = pd.read_csv(_csv_url("Courses"))
        transactions = pd.read_csv(_csv_url("Transactions"))
        teachers = pd.read_csv(_csv_url("Teachers"))
    except Exception as e:
        st.error(f"❌ Failed to load data from Google Sheets: {e}")
        st.stop()

    # --- Clean Users ---
    users.columns = users.columns.str.strip()
    users["UserID"] = users["UserID"].astype(str).str.strip()
    if "Age" in users.columns:
        users["Age"] = pd.to_numeric(users["Age"], errors="coerce")

    # --- Clean Courses ---
    courses.columns = courses.columns.str.strip()
    courses["CourseID"] = courses["CourseID"].astype(str).str.strip()
    for col in ["CourseRating", "CoursePrice", "CourseDuration"]:
        if col in courses.columns:
            courses[col] = pd.to_numeric(courses[col], errors="coerce")

    # --- Clean Transactions ---
    transactions.columns = transactions.columns.str.strip()
    transactions["UserID"] = transactions["UserID"].astype(str).str.strip()
    transactions["CourseID"] = transactions["CourseID"].astype(str).str.strip()
    transactions["Amount"] = pd.to_numeric(transactions["Amount"], errors="coerce").fillna(0)
    if "TransactionDate" in transactions.columns:
        transactions["TransactionDate"] = pd.to_datetime(
            transactions["TransactionDate"], errors="coerce"
        )

    # --- Clean Teachers ---
    teachers.columns = teachers.columns.str.strip()

    return users, courses, transactions, teachers
