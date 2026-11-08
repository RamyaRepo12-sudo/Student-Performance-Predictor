# Student Performance Prediction using Machine Learning

**Technologies:** Python, Pandas, NumPy, Scikit-learn (+ Matplotlib, Streamlit)

- Developed a predictive model on student academic data after data cleaning, preprocessing and Exploratory Data Analysis.
- Applied regression algorithms (Ridge, Random Forest, tuned Gradient Boosting) and evaluated them with MAE, RMSE, R² and cross-validation.

## What makes it suitable for real-world use
| Feature | Why it matters |
|---|---|
| **Two scenarios** (early warning / mid-term) | Predicting *before* any grade exists is what lets a school intervene in time |
| **Leak-free sklearn Pipeline** | Imputation, scaling, one-hot encoding fitted on training data only; model accepts raw rows |
| **Robust to messy input** | Missing values and missing columns are handled at prediction time |
| **Cross-validation + hyper-parameter tuning** | Honest model selection (nested CV for the tuned model) |
| **At-risk metrics** | Recall/precision for "predicted grade < 10", the decision that matters |
| **Prediction intervals + risk bands** | Every prediction has a 90% interval and a High / Medium / Low label |
| **Explainability** | Permutation importance shows which factors drive predictions |
| **Batch CLI + Streamlit dashboard** | Counsellors can score a whole class CSV or a single student |
| **Unit tests** | `pytest` checks cleaning, no-leakage, and prediction robustness |

## Project structure
```
data/            student_data.csv (synthetic, UCI-style schema), generate_data.py, sample_new_students.csv
src/             config.py, data_prep.py, eda.py, train.py, predict.py, report.py
models/          trained model bundles (.joblib)
reports/         report.html (open in browser), metrics.csv, figures/
app.py           Streamlit dashboard
tests/           pytest tests
```

## Run it
```bash
pip install -r requirements.txt
python data/generate_data.py          # optional: regenerate the dataset
python -m src.train                   # EDA + train + evaluate + save models + report
python -m src.predict --input data/sample_new_students.csv --scenario early --output predictions.csv
streamlit run app.py                  # interactive dashboard
pytest                                # run tests
```

## Using your own / real data
Replace `data/student_data.csv` with your institution's export. Keep the target column `G3` (final grade, 0-20),
or edit `src/config.py`. The public UCI "Student Performance" dataset works as a drop-in
(use `sep=";"` in `load_data`).

## Notes and limitations
- The bundled dataset is **synthetic**, so the numbers show the workflow, not real-world accuracy.
- Early-warning accuracy is naturally lower (R² about 0.4) because no past grades are used.
- Use predictions to **offer support**, never to penalise students; check for bias across groups before deployment.
