"""
t3t4_hybrid_cv.py  (TASKS T3 + T4)   -- leakage-safe version

L1 (overfitting/leakage) headline:
  (A) REPRODUCE the paper: 6 algorithms on the 55-point set.
      * in-sample (resubstitution) fit -> GB R2 ~ 1.000  (this IS the paper's 1.000:
        memorisation of 55 points, NOT a CV score).
      * row-wise LOO-CV -> GB RMSE ~ 1.88 (matches paper), Ensemble RMSE ~ 1.8.
  (B) REINTERPRET with leakage-free Leave-One-Mix-Out (GroupKFold by mix_id): the
      55 points are only 5 distinct binder mixes, so row-LOO leaks each mix's S-M
      curve. Under mix-CV the "perfect" GB collapses.
  (C) HYBRID physics-informed models (T3), with a FOLD-AWARE physics prior so the
      held-out mix's own strengths never leak through S_hyp:
        H1 residual learning : ML predicts f_c - S_hyp on top of hyperbolic prior
        H2 monotonic XGB     : monotone-increasing constraints on maturity + S_hyp
        H3 physics features  : paper features + hydration degree + fold-safe S_hyp
  (D) Learning curve (train-mix count vs error) to diagnose data saturation.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from config import OUT_DIR, CSV_ENC, RANDOM_STATE
from ml_utils import (make_models, metrics, cv_predict, loo_splits, group_splits,
                      FEATURES_PAPER, FEAT_MATURITY)
from physics_prior import PriorModel

MASTER = os.path.join(OUT_DIR, "master_phys.parquet")
pd.set_option("display.width", 200)

# hybrid feature set: paper features + deterministic hydration degree (+ S_hyp added per fold)
FEAT_HYB_BASE = FEATURES_PAPER + ["alpha_bazant", "alpha_fh"]


def Xy(df, feats):
    X = df[feats].astype(float)
    X = X.fillna(X.median()).values
    return X, df["f_c"].astype(float).values


def monotone_xgb(feats):
    inc = set(FEAT_MATURITY + ["alpha_bazant", "alpha_fh", "S_hyp"])
    cons = "(" + ",".join("1" if f in inc else "0" for f in feats) + ")"
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=2.0, min_child_weight=3,
        monotone_constraints=cons, random_state=RANDOM_STATE, verbosity=0)


def hybrid_oof(df, splits, kind):
    """Fold-aware hybrid OOF. Refits the physics prior on train rows each fold."""
    feats = FEAT_HYB_BASE + ["S_hyp"]
    y = df["f_c"].astype(float).values
    oof = np.full(len(df), np.nan)
    base = df[FEAT_HYB_BASE].astype(float)
    base = base.fillna(base.median())
    for tr, te in splits:
        pm = PriorModel().fit(df.iloc[tr])
        sh_tr = pm.predict(df.iloc[tr])
        sh_te = pm.predict(df.iloc[te])
        Xtr = np.column_stack([base.values[tr], sh_tr])
        Xte = np.column_stack([base.values[te], sh_te])
        if kind == "H1":                        # residual learning
            m = make_models()["GradientBoost"]
            m.fit(Xtr, y[tr] - sh_tr)
            oof[te] = sh_te + m.predict(Xte)
        elif kind == "H3":                      # physics features, GB on f_c
            m = make_models()["GradientBoost"]
            m.fit(Xtr, y[tr]); oof[te] = m.predict(Xte)
        elif kind == "H2":                      # monotone XGB
            m = monotone_xgb(feats)
            m.fit(Xtr, y[tr]); oof[te] = m.predict(Xte)
    return oof, y


def run_block(df, title, splits_fn):
    print("\n" + "-" * 78); print(title); print("-" * 78)
    rows = []
    X, y = Xy(df, FEATURES_PAPER)
    for name, mdl in make_models().items():
        oof = cv_predict(mdl, X, y, list(splits_fn()))
        rows.append(dict(model=name, kind="pure-ML", **metrics(y, oof)))
    # ensemble top-3
    top3 = ["GradientBoost", "Ridge_a0.1", "RandomForest"]
    preds = [cv_predict(make_models()[n], X, y, list(splits_fn())) for n in top3]
    rows.append(dict(model="Ensemble(top3)", kind="pure-ML", **metrics(y, np.mean(preds, 0))))
    # hybrids (fold-aware prior)
    for kind, label in [("H1", "H1_residual"), ("H3", "H3_GB+physfeat"), ("H2", "H2_monotonicXGB")]:
        oof, yy = hybrid_oof(df, list(splits_fn()), kind)
        rows.append(dict(model=label, kind="hybrid", **metrics(yy, oof)))
    res = pd.DataFrame(rows)
    print(res[["model", "kind", "RMSE", "MAE", "R2", "MAPE", "n"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    res.insert(0, "cv", title.split("]")[0].strip("[ "))
    return res


def insample_demo(df):
    """Resubstitution fit: GB memorises the 55 points -> R2 ~ 1.0 (the paper's 1.000)."""
    X, y = Xy(df, FEATURES_PAPER)
    m = make_models()["GradientBoost"]; m.fit(X, y)
    return metrics(y, m.predict(X))


def learning_curve(df):
    print("\n" + "-" * 78)
    print("Learning curve (leave random k mixes for train, test on rest; fold-safe prior)")
    print("-" * 78)
    mixes = list(df.mix_id.unique())
    base = df[FEAT_HYB_BASE].astype(float); base = base.fillna(base.median())
    y = df["f_c"].values
    rng = np.random.RandomState(RANDOM_STATE); out = []
    for k in range(2, len(mixes)):
        rmses = []
        for _ in range(10):
            tr_mix = set(rng.choice(mixes, k, replace=False))
            tr = df.mix_id.isin(tr_mix).values; te = ~tr
            if te.sum() == 0 or tr.sum() < 4:
                continue
            pm = PriorModel().fit(df[tr]); sh_tr = pm.predict(df[tr]); sh_te = pm.predict(df[te])
            m = make_models()["GradientBoost"]
            m.fit(np.column_stack([base.values[tr], sh_tr]), y[tr] - sh_tr)
            pred = sh_te + m.predict(np.column_stack([base.values[te], sh_te]))
            rmses.append(metrics(y[te], pred)["RMSE"])
        if rmses:
            out.append(dict(train_mixes=k, test_RMSE=np.mean(rmses), sd=np.std(rmses)))
    lc = pd.DataFrame(out)
    print(lc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lc.to_csv(os.path.join(OUT_DIR, "learning_curve.csv"), index=False, encoding=CSV_ENC)


def kfold_group_splits(groups, k=5, seed=RANDOM_STATE):
    """K-fold over GROUPS (whole mix-families held out together)."""
    groups = np.asarray(groups, dtype=object)
    uniq = np.array(list(pd.unique(groups)), dtype=object)
    rng = np.random.RandomState(seed); rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    for f in folds:
        te = np.where(np.isin(groups, f))[0]
        tr = np.where(~np.isin(groups, f))[0]
        if len(te) and len(tr):
            yield tr, te


def big_pool_block(df):
    """L3: does data scale + physics help NEW-mix generalisation on the 503 pool?"""
    print("\n" + "=" * 78)
    print("L3 EXPERIMENT - large literature pool (503), leave-mix-family-out (5-fold group)")
    print("=" * 78)
    pool = df[df.dataset == "lit503"].copy()
    pool = pool[pool.f_c.notna() & pool.t_eq.notna() & (pool.f_c > 0)].reset_index(drop=True)
    # drop extreme strengths (>100) & missing w/c to stabilise the pooled prior
    pool = pool[(pool.f_c <= 100) & pool.WC.notna()].reset_index(drop=True)
    print(f"pool: {len(pool)} rows, {pool.mix_family.nunique()} mix-families, "
          f"strength {pool.f_c.min():.1f}-{pool.f_c.max():.1f} MPa")
    splits = list(kfold_group_splits(pool.mix_family.values, k=5))
    rows = []
    X, y = Xy(pool, FEATURES_PAPER)
    for name in ["GradientBoost", "RandomForest", "Ridge_a0.1"]:
        oof = cv_predict(make_models()[name], X, y, splits)
        rows.append(dict(model=name, kind="pure-ML", **metrics(y, oof)))
    for kind, label in [("H1", "H1_residual"), ("H3", "H3_GB+physfeat"), ("H2", "H2_monotonicXGB")]:
        oof, yy = hybrid_oof(pool, splits, kind)
        rows.append(dict(model=label, kind="hybrid", **metrics(yy, oof)))
    res = pd.DataFrame(rows)
    print(res[["model", "kind", "RMSE", "MAE", "R2", "MAPE", "n"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    res.insert(0, "cv", "lit503_mixfamily5fold")
    res.to_csv(os.path.join(OUT_DIR, "t3t4_bigpool_results.csv"), index=False, encoding=CSV_ENC)
    gb = res[res.model == "GradientBoost"].RMSE.iloc[0]
    bh = res[res.kind == "hybrid"].RMSE.min()
    print(f"  -> pure GB RMSE={gb:.2f}  vs  best hybrid RMSE={bh:.2f}  "
          f"({'hybrid wins' if bh < gb else 'pure wins'})")
    return res


def main():
    print("=" * 78)
    print("TASKS T3+T4 - Physics-hybrid ML under leakage-free CV (fold-safe prior)")
    print("=" * 78)
    df = pd.read_parquet(MASTER)
    lit55 = df[df.dataset == "lit55"].reset_index(drop=True)
    print(f"lit55: {len(lit55)} rows, {lit55.mix_id.nunique()} distinct mixes.")

    ins = insample_demo(lit55)
    print(f"\n[In-sample resubstitution]  GradientBoost fits its own training data:")
    print(f"   R2 = {ins['R2']:.4f}   RMSE = {ins['RMSE']:.3f}   <-- THIS is the paper's "
          f"'R2=1.000' (memorisation, not generalisation)")

    all_res = [ins]
    r1 = run_block(lit55, "[row-LOO-CV]  (paper protocol; GB RMSE~1.88, Ensemble~1.8)",
                   lambda: loo_splits(len(lit55)))
    r2 = run_block(lit55, "[Leave-One-Mix-Out]  (leakage-free; GB should COLLAPSE)",
                   lambda: group_splits(lit55.mix_id.values))
    out = pd.concat([r1, r2], ignore_index=True)
    out.to_csv(os.path.join(OUT_DIR, "t3t4_cv_results.csv"), index=False, encoding=CSV_ENC)
    learning_curve(lit55)
    big_pool_block(df)

    def get(cv, model, metric):
        r = out[(out.cv.str.contains(cv)) & (out.model == model)]
        return float(r[metric].iloc[0]) if len(r) else np.nan
    print("\n" + "=" * 78)
    print("HEADLINE (L1): the 'perfect' GB is leakage/memorisation, not skill")
    print("=" * 78)
    print(f"  GB  in-sample R2 = {ins['R2']:.3f}  (memorisation)")
    print(f"  GB  R2  : row-LOO = {get('row-LOO','GradientBoost','R2'):.3f}"
          f"  ->  mix-CV = {get('Leave-One-Mix','GradientBoost','R2'):.3f}")
    print(f"  GB  RMSE: row-LOO = {get('row-LOO','GradientBoost','RMSE'):.2f}"
          f"  ->  mix-CV = {get('Leave-One-Mix','GradientBoost','RMSE'):.2f} MPa")
    hyb = out[(out.cv.str.contains('Leave-One-Mix')) & (out.kind == 'hybrid')]
    b = hyb.loc[hyb.RMSE.idxmin()]
    print(f"  BEST HYBRID (mix-CV, leakage-free): {b.model}  "
          f"RMSE={b.RMSE:.2f}  R2={b.R2:.3f}  MAPE={b.MAPE:.1f}%")
    print(f"  vs paper baseline Ensemble (row-LOO, leaky) RMSE=1.82")
    print("\nWritten -> t3t4_cv_results.csv, learning_curve.csv")


if __name__ == "__main__":
    main()
