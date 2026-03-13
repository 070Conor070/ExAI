# Student Performance — XAI Demo
**Dataset:** Adil Shamim · Student Performance & Learning Behavior · Kaggle 2025 · CC BY 4.0  
**14,003 students · 16 features · ExamScore target**

---

## Files

| File | What it does |
|------|-------------|
| `index.html` | Standalone browser demo — open directly, no server needed |
| `analysis.py` | Full Python pipeline: train model → real SHAP values → traversal CLI |
| `README.md` | This file |

---

## Browser demo (index.html)

Just double-click `index.html` — no install, no server.

**Tabs:**
- **Dataset explorer** — filter 14k students live, click any row for details, export filtered CSV
- **Student lookup** — enter any ID (1–14,003) or hit Random; see SHAP explanation per student
- **XAI / SHAP** — global feature importance bars + 4 charts (updates with active filters)
- **Python code** — download `analysis.py` directly from the browser
- **About** — dataset citation and all 16 feature descriptions

---

## Python pipeline (analysis.py)

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn shap matplotlib
```

### 2. Download the dataset
Go to: https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style  
Download `student_data.csv` and place it in the same folder as `analysis.py`.

### 3. Run
```bash
python analysis.py
```

### Output
| File | Description |
|------|-------------|
| `shap_global.png` | Bar chart of avg \|SHAP\| per feature |
| `shap_beeswarm.png` | Beeswarm plot — direction & magnitude of effects |
| `shap_student_N.png` | Waterfall plot for individual students |
| Console | RMSE / R², per-student SHAP values, traversal results |

### Traversal API
```python
from analysis import traverse

# High achievers
traverse(df, min_study=10, max_stress=4, grade="A")

# At-risk students
traverse(df, max_score=59)

# Custom sort
traverse(df, gender="Female", sort_by="MotivationLevel", top_n=30)
```

---

## Dataset features

| Feature | Type | Notes |
|---------|------|-------|
| StudentID | int | Unique ID |
| Age | int | 18–30 |
| Gender | cat | Male / Female |
| StudyHoursPerDay | float | Daily study time (hrs) |
| AttendanceRate | float | % classes attended |
| MotivationLevel | int | Self-rated 1–10 |
| StressLevel | int | Self-rated 1–10 |
| InternetAccess | cat | Yes / No |
| Resources | cat | High / Low |
| EduTechUsage | cat | Yes / No |
| OnlineCourses | int | Number completed |
| Discussions | int | Participation 0–10 |
| AssignmentCompletion | float | % completed |
| LearningStyle | cat | Visual / Auditory / Reading-Writing / Kinesthetic |
| ExamScore | float | 0–100 **(primary target)** |
| FinalGrade | cat | A / B / C / D / F |

---

## Citation
```
Adil Shamim. (2025). Student Performance and Learning Behavior Dataset.
Kaggle. https://kaggle.com/datasets/adilshamim8/student-performance-and-learning-style
License: CC BY 4.0
```
