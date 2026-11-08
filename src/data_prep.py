import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DATA_PATH, GRADE_COLS, TARGET

MAX_ABSENCES = 75   # domain rule: more than this is treated as a data-entry error and capped


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe to do BEFORE the train/test split
    (no statistics are learned from the data, so there is no leakage).
    Imputation / scaling / encoding live inside the sklearn Pipeline."""
    df = df.copy().drop_duplicates()
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].between(0, 20)]
    if "absences" in df:
        df["absences"] = df["absences"].clip(lower=0, upper=MAX_ABSENCES)
    return df.reset_index(drop=True)


def get_features(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    X = df.drop(columns=[TARGET], errors="ignore")
    if scenario == "early":
        X = X.drop(columns=GRADE_COLS, errors="ignore")
    return X


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
