"""Builds a single self-contained HTML report (images embedded)."""
import base64
import json

import pandas as pd

from .config import FIG_DIR, REPORTS_DIR, SCENARIOS


def img(name, w="100%"):
    p = FIG_DIR / name
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="max-width:{w}">'


def build_report():
    s = json.loads((REPORTS_DIR / "summary.json").read_text())
    m = pd.read_csv(REPORTS_DIR / "metrics.csv")
    eda = s["eda"]
    html = ["""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Student Performance Prediction - Report</title><style>
:root{--bg:#fff;--fg:#1e1b4b;--mut:#64748b;--card:#f5f3ff;--line:#ddd6fe;--acc:#4f46e5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f0e1f;--fg:#e8e7ff;--mut:#9aa0c3;--card:#1a1836;--line:#33305e;--acc:#a5b4fc}}
body{margin:0;padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0);background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;line-height:1.55}
main{max-width:980px;margin:auto;padding:24px 18px}h1{margin:.2em 0}h2{margin-top:2em;color:var(--acc)}
.kpis{display:flex;gap:12px;flex-wrap:wrap}.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;min-width:130px}
.kpi b{display:block;font-size:1.5em}.kpi span{color:var(--mut);font-size:.85em}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9em}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.note{background:var(--card);border-left:4px solid var(--acc);padding:10px 14px;border-radius:6px}</style></head><body><main>
<h1>Student Performance Prediction</h1>
<p style="color:var(--mut)">Python &middot; Pandas &middot; NumPy &middot; Scikit-learn &mdash; regression on student academic data</p>"""]
    html.append(f"""<div class="kpis"><div class="kpi"><b>{eda['raw_rows']}</b><span>raw records</span></div>
<div class="kpi"><b>{eda['rows']}</b><span>after cleaning</span></div>
<div class="kpi"><b>{eda['missing_cells']}</b><span>missing cells (imputed in pipeline)</span></div>
<div class="kpi"><b>{eda['pass_rate']:.0%}</b><span>pass rate (G3 &ge; 10)</span></div></div>""")
    html.append("<h2>1. Exploratory data analysis</h2><div class='grid'>"
                + img("eda_target.png") + img("eda_absences.png") + "</div>"
                + img("eda_boxplots.png") + "<div class='grid'>" + img("eda_corr.png", "100%") + "</div>")
    html.append("<h2>2. Model comparison (hold-out test set)</h2>")
    for sc, label in SCENARIOS.items():
        d = m[m.Scenario == sc].drop(columns="Scenario").round(3)
        html.append(f"<h3>{label}</h3><div class='wrap'>{d.to_html(index=False, border=0)}</div>"
                    f"<p>Selected model: <b>{s['best'][sc]}</b> (lowest cross-validated RMSE).</p>")
    html.append("""<div class="note"><b>Why two scenarios?</b> A model that uses G1/G2 is very accurate but only
useful late in the term. The <i>early warning</i> model uses only background and behaviour data, so it
can flag at-risk students <b>before</b> any grade exists, when intervention is most effective.</div>""")
    html.append("<h2>3. Diagnostics and key drivers</h2>")
    for sc, label in SCENARIOS.items():
        html.append(f"<h3>{label}</h3><div class='grid'>" + img(f"{sc}_actual_vs_pred.png")
                    + img(f"{sc}_residuals.png") + img(f"{sc}_importance.png") + "</div>")
    html.append("""<h2>4. Real-world use</h2><div class="note"><ul>
<li><b>Risk bands</b>: High (&lt;10), Medium (10-13), Low (&ge;13) predicted grade.</li>
<li><b>Uncertainty</b>: each prediction carries a 90% interval from held-out residuals.</li>
<li><b>No leakage</b>: imputation, scaling and encoding are fitted inside the Pipeline on training data only.</li>
<li><b>Ethics</b>: use predictions to offer support, never to penalise; audit for bias before deployment.</li></ul></div>
</main></body></html>""")
    (REPORTS_DIR / "report.html").write_text("\n".join(html), encoding="utf-8")
