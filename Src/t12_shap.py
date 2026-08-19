"""
t12_shap.py - post-hoc SHAP attribution of the augmented-feature hybrid (Fig. 11).

Reproduces the manuscript's Section 3.7 analysis on the CURRENT master dataset:
  * attribution set = the 677 unique records (master minus the duplicate lit55 subset)
  * physics prior S_hyp from the full-fit PriorModel (post-hoc interpretation, not CV)
  * learner = GradientBoostingRegressor(200 trees, depth 4, lr 0.02) on the 11 inputs
    (FEATURES_PAPER + alpha_bazant + alpha_fh + S_hyp)
  * TreeExplainer -> mean |SHAP| per feature (shap_importance.csv) + bar/beeswarm figure

Run:  .venv/Scripts/python.exe t12_shap.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import GradientBoostingRegressor

from config import OUT_DIR, FIG_DIR, CSV_ENC, RANDOM_STATE
from ml_utils import FEATURES_PAPER
from physics_prior import PriorModel

LABELS = {"S_hyp": "hyperbolic prior S_hyp", "log_time": "log(age)", "C": "cement",
          "WC": "W/C", "alpha_fh": "hydration alpha (F-H)", "M_NS": "Nurse-Saul M",
          "DD": "DD (degree-day)", "sqrt_DD": "sqrt(DD)",
          "alpha_bazant": "hydration alpha (Bazant)", "t_eq": "equiv. age t_eq", "RH": "RH"}


def main():
    df = pd.read_parquet(os.path.join(OUT_DIR, "master_phys.parquet"))
    df = df[df.dataset != "lit55"].reset_index(drop=True)   # 677 unique records
    print(f"attribution set: {len(df)} records")

    pm = PriorModel().fit(df)
    df["S_hyp"] = pm.predict(df)
    feats = FEATURES_PAPER + ["alpha_bazant", "alpha_fh", "S_hyp"]
    X = df[feats].astype(float)
    X = X.fillna(X.median())
    y = df["f_c"].astype(float).values

    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.02,
                                   random_state=RANDOM_STATE)
    gb.fit(X.values, y)

    ex = shap.TreeExplainer(gb)
    sv = ex.shap_values(X.values)
    imp = pd.DataFrame({"feature": feats,
                        "label": [LABELS.get(f, f) for f in feats],
                        "mean_abs_shap_MPa": np.abs(sv).mean(axis=0).round(3)})
    imp = imp.sort_values("mean_abs_shap_MPa", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False, encoding=CSV_ENC)
    print(imp.to_string(index=False))

    fig = plt.figure(figsize=(11, 4.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.barh(imp.label[::-1], imp.mean_abs_shap_MPa[::-1], color="#4C78A8")
    ax1.set_xlabel("mean |SHAP| (MPa)")
    ax1.set_title("(a) mean absolute SHAP attribution")
    for yy, v in enumerate(imp.mean_abs_shap_MPa[::-1]):
        ax1.text(v, yy, f" {v:.2f}", va="center", fontsize=7)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.sca(ax2)
    shap.summary_plot(sv, X, feature_names=[LABELS.get(f, f) for f in feats],
                      show=False, plot_size=None, max_display=11)
    ax2.set_title("(b) SHAP summary (beeswarm)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_shap.png"), dpi=200, bbox_inches="tight")
    print("written -> shap_importance.csv, fig7_shap.png")


if __name__ == "__main__":
    main()
