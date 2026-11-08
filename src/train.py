"""Train, tune, evaluate and save models for both real-world scenarios.
Run:  python -m src.train
"""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import FIG_DIR, MODELS_DIR, PASS_MARK, REPORTS_DIR, SCENARIOS, SEED, TARGET
from .data_prep import build_preprocessor, clean_data, get_features, load_data
from .eda import run_eda

warnings.filterwarnings("ignore")
C = "#4f46e5"


def candidates(X):
    def pipe(model):
        return Pipeline([("prep", build_preprocessor(X)), ("model", model)])
    tuned = RandomizedSearchCV(
        pipe(GradientBoostingRegressor(random_state=SEED)),
        {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
         "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0]},
        n_iter=8, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=-1)
    return {
        "Ridge Regression": pipe(Ridge(alpha=1.0)),
        "Random Forest": pipe(RandomForestRegressor(n_estimators=250, min_samples_leaf=2,
                                                    random_state=SEED, n_jobs=-1)),
        "Gradient Boosting (tuned)": tuned,
    }


def evaluate(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, 20)
    at_risk_true, at_risk_pred = y_true < PASS_MARK, y_pred < PASS_MARK
    tp = int((at_risk_true & at_risk_pred).sum())
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "AtRisk_Recall": tp / max(int(at_risk_true.sum()), 1),
        "AtRisk_Precision": tp / max(int(at_risk_pred.sum()), 1),
    }


def run_scenario(df, scenario):
    X, y = get_features(df, scenario), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    kf = KFold(5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, est in candidates(X_tr).items():
        cv = -cross_val_score(est, X_tr, y_tr, cv=kf, scoring="neg_root_mean_squared_error")
        est.fit(X_tr, y_tr)
        fitted[name] = est.best_estimator_ if hasattr(est, "best_estimator_") else est
        m = evaluate(y_te.values, fitted[name].predict(X_te))
        rows.append({"Scenario": scenario, "Model": name, "CV_RMSE": cv.mean(), **m})
    results = pd.DataFrame(rows)
    best_name = results.sort_values("CV_RMSE").iloc[0]["Model"]
    best = fitted[best_name]
    pred = np.clip(best.predict(X_te), 0, 20)

    # residual-based 90% prediction interval (honest uncertainty for real-world use)
    resid = y_te.values - pred
    q_lo, q_hi = np.quantile(resid, [0.05, 0.95])

    imp = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=SEED,
                                 scoring="neg_root_mean_squared_error")
    imp_df = pd.Series(imp.importances_mean, index=X_te.columns).sort_values(ascending=False)

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(y_te, pred, s=14, alpha=.5, color=C)
    ax.plot([0, 20], [0, 20], "r--"); ax.set(xlabel="Actual G3", ylabel="Predicted G3",
                                             title=f"Actual vs predicted ({scenario})")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{scenario}_actual_vs_pred.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.hist(resid, bins=25, color=C, edgecolor="white")
    ax.set(xlabel="Residual (actual - predicted)", title=f"Residuals ({scenario})")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{scenario}_residuals.png"); plt.close(fig)

    top = imp_df.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top.index, top.values, color=C)
    ax.set(xlabel="Increase in RMSE when shuffled", title=f"Top drivers ({scenario})")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{scenario}_importance.png"); plt.close(fig)

    bundle = {
        "model": best, "scenario": scenario, "best_name": best_name,
        "features": X.columns.tolist(),
        "defaults": {c: (X[c].median() if pd.api.types.is_numeric_dtype(X[c]) else X[c].mode()[0]) for c in X.columns},
        "interval": (float(q_lo), float(q_hi)),
        "metrics": results[results.Model == best_name].iloc[0].to_dict(),
    }
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / f"model_{scenario}.joblib")
    return results, imp_df.head(10).round(4).to_dict(), best_name


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    df = clean_data(raw)
    print(f"Loaded {len(raw)} rows -> {len(df)} after cleaning")
    eda = run_eda(df)
    eda["raw_rows"], eda["duplicates_removed"] = len(raw), len(raw) - len(df)

    all_res, importance, best = [], {}, {}
    for sc in SCENARIOS:
        res, imp, b = run_scenario(df, sc)
        all_res.append(res); importance[sc] = imp; best[sc] = b
        print(f"\n== {SCENARIOS[sc]} ==\n{res.round(3).to_string(index=False)}\nBest: {b}")
    results = pd.concat(all_res)
    results.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    (REPORTS_DIR / "summary.json").write_text(json.dumps(
        {"eda": eda, "importance": importance, "best": best}, indent=2, default=float))

    from .report import build_report
    build_report()
    print("\nSaved models/, reports/metrics.csv, reports/report.html")


if __name__ == "__main__":
    main()
