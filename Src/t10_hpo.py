"""
t10_hpo.py  (503-pool hyperparameter optimisation, leakage-safe)

Tune the residual Gradient-Boosting learner of the physics hybrid (H1) on the large
503-mix literature pool under LEAVE-MIX-FAMILY-OUT group CV. Key efficiency point: the
fold-safe physics prior S_hyp depends only on the CV split, NOT on the GB hyper-parameters
-> we precompute (X, S_hyp, residual) per fold once, then sweep hyper-parameters cheaply.

Reported: default GB vs tuned GB (RMSE/R2/MAE), and the winning configuration.
"""
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from config import OUT_DIR, CSV_ENC, RANDOM_STATE
from ml_utils import metrics
from physics_prior import PriorModel
from t3t4_hybrid_cv import kfold_group_splits, FEAT_HYB_BASE

MASTER = os.path.join(OUT_DIR, "master_phys.parquet")
pd.set_option("display.width", 200)


def main():
    print("=" * 78)
    print("TASK T10 - 503-pool hyperparameter optimisation (leave-mix-family-out)")
    print("=" * 78)
    df = pd.read_parquet(MASTER)
    pool = df[df.dataset == "lit503"].copy()
    pool = pool[pool.f_c.notna() & pool.t_eq.notna() & (pool.f_c > 0) &
                (pool.f_c <= 100) & pool.WC.notna()].reset_index(drop=True)
    y = pool["f_c"].values
    base = pool[FEAT_HYB_BASE].astype(float)
    base = base.fillna(base.median())
    splits = list(kfold_group_splits(pool.mix_family.values, k=5))
    print(f"pool {len(pool)} rows, {pool.mix_family.nunique()} families, 5 group folds")

    # ---- precompute per-fold prior / features / residual targets ----
    folds = []
    for tr, te in splits:
        pm = PriorModel().fit(pool.iloc[tr])
        sh_tr, sh_te = pm.predict(pool.iloc[tr]), pm.predict(pool.iloc[te])
        Xtr = np.column_stack([base.values[tr], sh_tr])
        Xte = np.column_stack([base.values[te], sh_te])
        folds.append((tr, te, Xtr, Xte, sh_tr, sh_te))

    def oof_rmse(params):
        oof = np.full(len(pool), np.nan)
        for tr, te, Xtr, Xte, sh_tr, sh_te in folds:
            m = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
            m.fit(Xtr, y[tr] - sh_tr)
            oof[te] = sh_te + m.predict(Xte)
        return oof

    # ---- default ----
    default = dict(n_estimators=100, max_depth=3, learning_rate=0.1)
    m_def = metrics(y, oof_rmse(default))
    print(f"\ndefault GB {default}: RMSE {m_def['RMSE']:.2f}  R2 {m_def['R2']:.3f}  MAE {m_def['MAE']:.2f}")

    # ---- randomized grid search ----
    grid = dict(n_estimators=[200, 400, 600], max_depth=[2, 3, 4],
                learning_rate=[0.02, 0.03, 0.05, 0.1], subsample=[0.7, 0.85, 1.0],
                min_samples_leaf=[1, 3, 5])
    combos = list(itertools.product(*grid.values()))
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(combos), size=min(40, len(combos)), replace=False)
    keys = list(grid.keys())
    results = []
    for i in idx:
        params = dict(zip(keys, combos[i]))
        mm = metrics(y, oof_rmse(params))
        results.append(dict(**params, RMSE=mm["RMSE"], R2=mm["R2"], MAE=mm["MAE"]))
    res = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print(f"\nsearched {len(res)} configs. Top 5:")
    print(res.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best = res.iloc[0]
    print(f"\nBEST: RMSE {best.RMSE:.2f} (default {m_def['RMSE']:.2f}, "
          f"{100*(m_def['RMSE']-best.RMSE)/m_def['RMSE']:+.1f}%), "
          f"R2 {best.R2:.3f} (default {m_def['R2']:.3f})")
    res.to_csv(os.path.join(OUT_DIR, "hpo_results.csv"), index=False, encoding=CSV_ENC)
    pd.DataFrame([{"config": "default", **default, "RMSE": m_def["RMSE"], "R2": m_def["R2"]},
                  {"config": "tuned", **{k: best[k] for k in keys}, "RMSE": best.RMSE, "R2": best.R2}]
                 ).to_csv(os.path.join(OUT_DIR, "hpo_summary.csv"), index=False, encoding=CSV_ENC)
    print("Written -> hpo_results.csv, hpo_summary.csv")


if __name__ == "__main__":
    main()
