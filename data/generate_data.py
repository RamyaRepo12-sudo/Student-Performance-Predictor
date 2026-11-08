"""Generate a realistic synthetic student dataset (schema follows the UCI
'Student Performance' dataset). Includes missing values, duplicates and
outliers on purpose so the cleaning step has real work to do.

Replace data/student_data.csv with your own institution's export to use
real data - just keep the column names (or edit src/config.py)."""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 1000

def choice(options, p=None):
    return rng.choice(options, n, p=p)

df = pd.DataFrame({
    "school": choice(["GP", "MS"], [0.75, 0.25]),
    "sex": choice(["F", "M"]),
    "age": rng.integers(15, 20, n),
    "address": choice(["U", "R"], [0.7, 0.3]),
    "famsize": choice(["GT3", "LE3"], [0.7, 0.3]),
    "Pstatus": choice(["T", "A"], [0.88, 0.12]),
    "Medu": rng.integers(0, 5, n),
    "Fedu": rng.integers(0, 5, n),
    "traveltime": rng.choice([1, 2, 3, 4], n, p=[0.55, 0.3, 0.1, 0.05]),
    "studytime": rng.choice([1, 2, 3, 4], n, p=[0.25, 0.45, 0.2, 0.1]),
    "failures": rng.choice([0, 1, 2, 3], n, p=[0.78, 0.13, 0.06, 0.03]),
    "schoolsup": choice(["yes", "no"], [0.12, 0.88]),
    "famsup": choice(["yes", "no"], [0.6, 0.4]),
    "paid": choice(["yes", "no"], [0.4, 0.6]),
    "activities": choice(["yes", "no"]),
    "higher": choice(["yes", "no"], [0.9, 0.1]),
    "internet": choice(["yes", "no"], [0.8, 0.2]),
    "romantic": choice(["yes", "no"], [0.35, 0.65]),
    "famrel": rng.choice([1, 2, 3, 4, 5], n, p=[0.04, 0.06, 0.17, 0.48, 0.25]),
    "freetime": rng.integers(1, 6, n),
    "goout": rng.integers(1, 6, n),
    "Dalc": rng.choice([1, 2, 3, 4, 5], n, p=[0.7, 0.15, 0.08, 0.04, 0.03]),
    "Walc": rng.choice([1, 2, 3, 4, 5], n, p=[0.35, 0.25, 0.2, 0.12, 0.08]),
    "health": rng.integers(1, 6, n),
})
df["absences"] = np.clip(rng.poisson(4 + 2 * df.failures + 0.8 * (df.Walc - 1)), 0, 60)

ability = rng.normal(11.5, 2.3, n)
signal = (
    ability
    + 0.9 * (df.studytime - 2)
    - 1.6 * df.failures
    - 0.12 * df.absences
    + 0.25 * (df.Medu + df.Fedu - 4)
    + np.where(df.higher == "yes", 1.0, -1.5)
    - 0.25 * (df.Walc - 1) - 0.2 * (df.goout - 3)
    + np.where(df.internet == "yes", 0.4, 0)
    + np.where(df.paid == "yes", 0.3, 0)
    - 0.15 * (df.age - 16)
)
df["G1"] = np.clip(np.round(signal + rng.normal(0, 1.4, n)), 0, 20)
df["G2"] = np.clip(np.round(0.75 * df.G1 + 0.25 * signal + rng.normal(0, 1.1, n)), 0, 20)
df["G3"] = np.clip(np.round(0.8 * df.G2 + 0.2 * signal + rng.normal(0, 1.0, n)), 0, 20)
drop = (df.G3 > 0) & (rng.random(n) < 0.02)      # a few students who dropped out -> G3 = 0
df.loc[drop, "G3"] = 0

# --- inject real-world messiness ---
for col in ["studytime", "absences", "Medu", "higher", "internet", "G1"]:
    df.loc[rng.random(n) < 0.02, col] = np.nan
df.loc[rng.choice(n, 6, replace=False), "absences"] = rng.integers(90, 160, 6)   # data-entry outliers
df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)           # duplicates

out = Path(__file__).parent / "student_data.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows x {df.shape[1]} cols -> {out}")
