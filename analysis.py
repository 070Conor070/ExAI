"""
Student Performance — XAI Analysis
===================================
Dataset : Student Performance & Learning Behavior Dataset
Source  : https://kaggle.com/datasets/adilshamim8/student-performance-and-learning-style
Author  : Adil Shamim · Kaggle · 2025 · CC BY 4.0

Setup
-----
pip install pandas numpy scikit-learn shap matplotlib

Download the CSV from Kaggle and save as: student_data.csv (same folder as this file)
Then run:  python analysis.py
"""

import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "student_data.csv"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"\n[ERROR] Could not find '{CSV_PATH}'.")
    print("  Download from: https://kaggle.com/datasets/adilshamim8/student-performance-and-learning-style")
    print("  Save as 'student_data.csv' in the same folder as this script.\n")
    sys.exit(1)

print(f"✓ Loaded {len(df):,} rows  |  {df.shape[1]} columns")
print(df.head(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2. Clean & encode
# ─────────────────────────────────────────────────────────────────────────────
# Normalise column names (handle minor naming differences across versions)
df.columns = [c.strip().replace(" ", "") for c in df.columns]

FEATURE_MAP = {
    "StudyHoursPerDay": ["StudyHoursPerDay","StudyHours","Study_Hours"],
    "AttendanceRate":   ["AttendanceRate","Attendance","Attendance_Rate"],
    "MotivationLevel":  ["MotivationLevel","Motivation","Motivation_Level"],
    "StressLevel":      ["StressLevel","Stress","Stress_Level"],
    "InternetAccess":   ["InternetAccess","Internet","Internet_Access"],
    "Resources":        ["Resources","ResourceAccess","Resource_Access"],
    "EduTechUsage":     ["EduTechUsage","EduTech","EduTech_Usage"],
    "OnlineCourses":    ["OnlineCourses","Online_Courses"],
    "Discussions":      ["Discussions","Discussion"],
    "AssignmentCompletion":["AssignmentCompletion","Assignment_Completion","AssignmentComp"],
    "LearningStyle":    ["LearningStyle","Learning_Style"],
    "Age":              ["Age"],
    "Gender":           ["Gender"],
    "ExamScore":        ["ExamScore","Exam_Score","FinalExamScore"],
    "FinalGrade":       ["FinalGrade","Final_Grade","Grade"],
    "StudentID":        ["StudentID","Student_ID","ID"],
}

# Rename columns to canonical names
rename = {}
for canonical, alts in FEATURE_MAP.items():
    for alt in alts:
        if alt in df.columns and canonical not in df.columns:
            rename[alt] = canonical
            break
if rename:
    df = df.rename(columns=rename)

cat_cols = ["Gender", "InternetAccess", "Resources", "EduTechUsage", "LearningStyle"]
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

FEATURES = [f for f in [
    "Age","Gender","StudyHoursPerDay","AttendanceRate","MotivationLevel",
    "StressLevel","InternetAccess","Resources","EduTechUsage",
    "OnlineCourses","Discussions","AssignmentCompletion","LearningStyle"
] if f in df.columns]

TARGET = "ExamScore"
assert TARGET in df.columns, f"Could not find target column '{TARGET}'. Check your CSV column names."

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET]

print(f"\n✓ Using {len(FEATURES)} features  |  {len(df):,} complete rows after cleaning")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / test split & model
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
r2    = r2_score(y_test, preds)
print(f"\n── Model performance ──────────────────────────────")
print(f"  Test RMSE : {rmse:.2f}")
print(f"  Test R²   : {r2:.4f}")
print(f"──────────────────────────────────────────────────")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP — global importance bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("\n⏳ Computing SHAP values (this may take ~30 s for 14 k rows)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test)

fig, ax = plt.subplots(figsize=(9, 5))
shap.plots.bar(shap_values, max_display=len(FEATURES), show=False, ax=ax)
ax.set_title("Global SHAP Feature Importance — Student Exam Score", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150)
plt.close()
print("  Saved → shap_global.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP beeswarm
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.beeswarm(shap_values, max_display=len(FEATURES), show=False, ax=ax)
ax.set_title("SHAP Beeswarm — Direction & Magnitude of Each Feature", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)
plt.close()
print("  Saved → shap_beeswarm.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Single-student SHAP explanation
# ─────────────────────────────────────────────────────────────────────────────
def explain_student(row_index_in_test: int = 0):
    """Print and plot SHAP explanation for a single student."""
    student = X_test.iloc[[row_index_in_test]]
    sv      = explainer(student)
    pred    = model.predict(student)[0]
    actual  = y_test.iloc[row_index_in_test]
    sid     = X_test.index[row_index_in_test]

    print(f"\n── Student #{sid} ──────────────────────────────────")
    print(f"  Predicted score : {pred:.1f}")
    print(f"  Actual score    : {actual:.1f}")
    print(f"\n  SHAP contributions (sorted by |impact|):")
    pairs = sorted(zip(FEATURES, sv.values[0]), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in pairs:
        bar  = ("█" * int(abs(val) / 0.5)).ljust(20)
        sign = "▲ +" if val > 0 else "▼  "
        print(f"  {sign}{abs(val):5.2f}  {bar}  {feat}")
    print(f"──────────────────────────────────────────────────")

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(sv[0], show=False)
    plt.title(f"Student #{sid} — SHAP Waterfall (predicted={pred:.1f})", fontsize=12)
    plt.tight_layout()
    fname = f"shap_student_{sid}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved → {fname}")

explain_student(0)   # first test-set student
explain_student(1)   # second test-set student

# ─────────────────────────────────────────────────────────────────────────────
# 7. Dataset traversal
# ─────────────────────────────────────────────────────────────────────────────
def traverse(
    dataframe,
    min_study:   float = None,
    max_stress:  int   = None,
    grade:       str   = None,
    gender:      str   = None,
    min_score:   float = None,
    max_score:   float = None,
    top_n:       int   = 20,
    sort_by:     str   = "ExamScore",
    ascending:   bool  = False,
    verbose:     bool  = True,
) -> pd.DataFrame:
    """
    Filter the student dataset and return matching rows.

    Parameters
    ----------
    min_study   : minimum StudyHoursPerDay
    max_stress  : maximum StressLevel (1-10)
    grade       : FinalGrade  ('A','B','C','D','F')
    gender      : 'Male' or 'Female'  (pre-encoding)
    min_score   : minimum ExamScore
    max_score   : maximum ExamScore
    top_n       : rows to display
    sort_by     : column to sort results by
    ascending   : sort direction
    verbose     : print results to console

    Returns
    -------
    pd.DataFrame  filtered & sorted subset
    """
    # Work on original (re-loaded) df so categoricals are readable
    raw = pd.read_csv(CSV_PATH)
    raw.columns = [c.strip().replace(" ", "") for c in raw.columns]
    if rename:
        raw = raw.rename(columns=rename)

    result = raw.copy()
    if min_study  is not None: result = result[result["StudyHoursPerDay"] >= min_study]
    if max_stress is not None: result = result[result["StressLevel"]      <= max_stress]
    if min_score  is not None: result = result[result[TARGET]             >= min_score]
    if max_score  is not None: result = result[result[TARGET]             <= max_score]
    if grade      is not None: result = result[result["FinalGrade"]       == grade.upper()]
    if gender     is not None: result = result[result["Gender"]           == gender]

    result = result.sort_values(sort_by, ascending=ascending)

    display_cols = [c for c in ["StudentID","Age","Gender","StudyHoursPerDay",
                                "StressLevel","MotivationLevel","AttendanceRate",
                                "ExamScore","FinalGrade"] if c in result.columns]

    if verbose:
        print(f"\n{'─'*56}")
        print(f"  Traversal result : {len(result):,} students match")
        print(f"  Showing top {min(top_n, len(result))} sorted by {sort_by} ({'asc' if ascending else 'desc'})")
        print(f"{'─'*56}")
        print(result[display_cols].head(top_n).to_string(index=False))
        print(f"{'─'*56}\n")

    return result


print("\n── Example traversals ─────────────────────────────────────────────────")

print("\n[1] High achievers: study ≥ 10 h, stress ≤ 4, grade A")
high = traverse(df, min_study=10, max_stress=4, grade="A")

print("\n[2] At-risk students: score < 60, stress ≥ 8")
risk = traverse(df, max_score=59, min_score=0)
risk2 = risk[risk["StressLevel"] >= 8] if "StressLevel" in risk.columns else risk
print(f"  {len(risk2):,} at-risk students with high stress")
print(risk2[["StudentID","StudyHoursPerDay","StressLevel","ExamScore","FinalGrade"]].head(10).to_string(index=False))

print("\n[3] Female students with highest motivation")
traverse(df, gender="Female", sort_by="MotivationLevel")

print("\n✓ All done! Check the current folder for shap_*.png plots.")
