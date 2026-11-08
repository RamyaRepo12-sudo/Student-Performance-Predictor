import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR, PASS_MARK, TARGET

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
C = "#4f46e5"


def run_eda(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"rows": len(df), "cols": df.shape[1],
               "missing_cells": int(df.isna().sum().sum()),
               "pass_rate": float((df[TARGET] >= PASS_MARK).mean())}

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(df[TARGET], bins=21, color=C, edgecolor="white")
    ax.axvline(PASS_MARK, color="crimson", ls="--", label=f"Pass mark ({PASS_MARK})")
    ax.set(title="Distribution of final grade (G3)", xlabel="G3", ylabel="Students"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_target.png"); plt.close(fig)

    corr = df.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, fraction=0.04); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_corr.png"); plt.close(fig)

    c = corr[TARGET].drop(TARGET)
    summary["top_corr"] = c.reindex(c.abs().sort_values(ascending=False).index[:6]).round(3).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, col in zip(axes, ["studytime", "failures", "higher"]):
        groups = sorted(df[col].dropna().unique())
        ax.boxplot([df.loc[df[col] == g, TARGET] for g in groups], tick_labels=[str(g) for g in groups],
                   patch_artist=True, boxprops=dict(facecolor="#c7d2fe"))
        ax.set(title=f"G3 by {col}", xlabel=col, ylabel="G3")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_boxplots.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(df["absences"], df[TARGET], s=10, alpha=.4, color=C)
    z = np.polyfit(df["absences"].fillna(0), df[TARGET], 1)
    xs = np.linspace(0, df["absences"].max(), 50); ax.plot(xs, np.polyval(z, xs), color="crimson")
    ax.set(title="Absences vs final grade", xlabel="Absences", ylabel="G3")
    fig.tight_layout(); fig.savefig(FIG_DIR / "eda_absences.png"); plt.close(fig)
    return summary
