"""Streamlit dashboard:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.config import SCENARIOS
from src.predict import align_columns, ensure_models, load_bundle, predict_df

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Predictor")


@st.cache_resource(show_spinner="First run: training models (about a minute)...")
def get_bundle(scenario):
    ensure_models()
    return load_bundle(scenario)


sc = st.sidebar.radio("Scenario", list(SCENARIOS), format_func=SCENARIOS.get)
bundle = get_bundle(sc)
m = bundle["metrics"]
st.sidebar.metric("Model", bundle["best_name"])
st.sidebar.metric("Test MAE (grade points)", f"{m['MAE']:.2f}")
st.sidebar.metric("R²", f"{m['R2']:.2f}")

tab1, tab2 = st.tabs(["Single student", "Batch (CSV)"])

with tab1:
    d = bundle["defaults"]
    c1, c2, c3 = st.columns(3)
    row = dict(d)
    row["studytime"] = c1.slider("Weekly study time (1=<2h … 4=>10h)", 1, 4, int(d["studytime"]))
    row["failures"] = c1.slider("Past class failures", 0, 3, int(d["failures"]))
    row["absences"] = c1.slider("Absences", 0, 75, int(d["absences"]))
    row["age"] = c2.slider("Age", 15, 22, int(d["age"]))
    row["Medu"] = c2.slider("Mother's education (0-4)", 0, 4, int(d["Medu"]))
    row["Fedu"] = c2.slider("Father's education (0-4)", 0, 4, int(d["Fedu"]))
    row["goout"] = c3.slider("Going out (1-5)", 1, 5, int(d["goout"]))
    row["Walc"] = c3.slider("Weekend alcohol (1-5)", 1, 5, int(d["Walc"]))
    row["higher"] = c3.selectbox("Wants higher education", ["yes", "no"])
    row["internet"] = c3.selectbox("Internet at home", ["yes", "no"])
    row["paid"] = c2.selectbox("Paid extra classes", ["no", "yes"])
    if sc == "midterm":
        row["G1"] = c1.slider("G1 (period 1 grade)", 0, 20, 10)
        row["G2"] = c1.slider("G2 (period 2 grade)", 0, 20, 10)
    res = predict_df(pd.DataFrame([row]), sc).iloc[0]
    st.subheader(f"Predicted final grade: {res.predicted_G3} / 20")
    st.write(f"90% interval: **{res.interval_low} – {res.interval_high}**")
    {"High": st.error, "Medium": st.warning, "Low": st.success}[res.risk_level](
        f"Risk level: {res.risk_level}")

with tab2:
    up = st.file_uploader("Upload a CSV of students", type="csv")
    if up:
        data = pd.read_csv(up)
        _, missing, extra = align_columns(data, bundle)
        if missing:
            st.warning(f"{len(missing)} column(s) not in your file were filled with typical values: "
                       + ", ".join(missing))
        if extra:
            st.info("Ignored columns not used by the model: " + ", ".join(extra))
        out = predict_df(data, sc)
        st.dataframe(out, use_container_width=True)
        st.bar_chart(out["risk_level"].value_counts())
        st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
