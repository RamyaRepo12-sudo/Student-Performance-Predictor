from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_data.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

TARGET = "G3"                 # final grade (0-20)
GRADE_COLS = ["G1", "G2"]     # period grades, only known later in the term
PASS_MARK = 10                # below this = "at risk"
SEED = 42

# Two real-world scenarios:
#  early   -> predict BEFORE any grade exists (enrolment / week 1), maximum time to intervene
#  midterm -> predict after G1 and G2 are known, highest accuracy
SCENARIOS = {
    "early": "Early warning (no grades available yet)",
    "midterm": "Mid-term (G1 and G2 available)",
}


def risk_level(grade: float) -> str:
    """Convert a predicted final grade into an action-oriented risk band."""
    if grade < PASS_MARK:
        return "High"
    if grade < 13:
        return "Medium"
    return "Low"
