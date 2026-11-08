import numpy as np
import pandas as pd

from src.data_prep import clean_data, get_features, load_data
from src.predict import predict_df


def test_clean_removes_dupes_and_caps_outliers():
    df = load_data()
    out = clean_data(df)
    assert len(out) < len(df)
    assert out["absences"].max() <= 75
    assert out["G3"].between(0, 20).all()


def test_early_scenario_has_no_grade_features():
    X = get_features(clean_data(load_data()), "early")
    assert "G1" not in X and "G2" not in X and "G3" not in X


def test_prediction_handles_missing_values_and_columns():
    df = clean_data(load_data()).head(5)
    X = get_features(df, "early").copy()
    X.loc[0, "studytime"] = np.nan
    X = X.drop(columns=["health"])
    out = predict_df(X, "early")
    assert out["predicted_G3"].between(0, 20).all()
    assert set(out["risk_level"]) <= {"High", "Medium", "Low"}


def test_missing_categorical_columns_and_unseen_values():
    df = get_features(clean_data(load_data()), "early").head(3).drop(columns=["school", "sex"])
    df["address"] = "unknown-value"
    out = predict_df(df, "early")
    assert out["predicted_G3"].between(0, 20).all()


def test_models_self_heal(tmp_path, monkeypatch):
    import shutil
    from src import config, predict
    backup = tmp_path / "bak"
    shutil.copytree(config.MODELS_DIR, backup)
    shutil.rmtree(config.MODELS_DIR)
    try:
        out = predict.predict_df(get_features(clean_data(load_data()), "early").head(2), "early")
        assert len(out) == 2
    finally:
        shutil.rmtree(config.MODELS_DIR, ignore_errors=True)
        shutil.copytree(backup, config.MODELS_DIR)
