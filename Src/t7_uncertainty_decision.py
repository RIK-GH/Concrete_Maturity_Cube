"""
t7_uncertainty_decision.py  (TASK T7)

From point predictions to RISK-BASED formwork-removal decisions.

(1) UNCERTAINTY: split-conformal prediction intervals on top of the hybrid H1
    predictor. Nonconformity = |residual| from leakage-free group-CV on the 503
    pool; the (1-alpha) empirical quantile gives a distribution-free interval with
    guaranteed marginal coverage. We also fit quantile Gradient Boosting (5/50/95%)
    for an asymmetric comparison. Reported: PICP (coverage) and mean interval width.

(2) DECISION LAYER: for the slab, propagate the interval into an exceedance
    probability against code thresholds, as a function of equivalent age:
        P(S >= 14 MPa)            KCS 14 20 12 absolute
        P(S >= 2/3 f'c)           KCS 14 20 12 relative
        P(S >= 0.70 f'c)          ACI 318/347R formwork
    A Gaussian around the point prediction with sigma from the conformal half-width
    yields the probability; the recommended strike age is the earliest age at which
    P >= 0.95 for the governing criterion.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from config import OUT_DIR, CSV_ENC, RANDOM_STATE, CODE_THRESHOLDS
from ml_utils import make_models, metrics
from physics_prior import PriorModel
from t3t4_hybrid_cv import kfold_group_splits

MASTER = os.path.join(OUT_DIR, "master_phys.parquet")
FEAT = ["t_eq", "log_teq", "sqrt_teq", "WC", "C", "RH"]
pd.set_option("display.width", 200)


def build_feats(df, te):
    te = np.asarray(te, float)
    return pd.DataFrame({
        "t_eq": te, "log_teq": np.log(np.clip(te, 1e-3, None)),
        "sqrt_teq": np.sqrt(np.clip(te, 0, None)),
        "WC": df["WC"].values, "C": df["C"].values, "RH": df["RH"].values},
        index=df.index)


def main():
    print("=" * 78)
    print("TASK T7 - Uncertainty quantification + risk-based decision layer")
    print("=" * 78)
    df = pd.read_parquet(MASTER)
    pool = df[df.dataset.isin(["lit503", "lit119"])].copy()
    pool = pool[pool.f_c.notna() & pool.t_eq.notna() & (pool.f_c > 0) &
                (pool.f_c <= 60) & pool.WC.between(0.35, 0.72)].reset_index(drop=True)
    for c in ["C", "RH"]:
        pool[c] = pool[c].fillna(pool[c].median())
    y = pool["f_c"].values

    # ---- leakage-free group-CV OOF residuals for the hybrid H1 ----
    splits = list(kfold_group_splits(pool.mix_family.values, k=5))
    base = build_feats(pool, pool["t_eq"].values)
    med = base.median()
    oof = np.full(len(pool), np.nan)
    for tr, te in splits:
        pm = PriorModel().fit(pool.iloc[tr])
        sh_tr, sh_te = pm.predict(pool.iloc[tr]), pm.predict(pool.iloc[te])
        Xtr = np.column_stack([base.iloc[tr].fillna(med).values, sh_tr])
        Xte = np.column_stack([base.iloc[te].fillna(med).values, sh_te])
        m = make_models()["GradientBoost"]; m.fit(Xtr, y[tr] - sh_tr)
        oof[te] = sh_te + m.predict(Xte)
    resid = y - oof
    print(f"\nHybrid H1 group-CV: {metrics(y, oof)}")

    # ---- split-conformal interval (distribution-free) ----
    for alpha in [0.10, 0.20]:
        q = np.quantile(np.abs(resid), 1 - alpha)
        lo, hi = oof - q, oof + q
        picp = np.mean((y >= lo) & (y <= hi))
        print(f"  conformal {int((1-alpha)*100)}% : half-width={q:5.2f} MPa | "
              f"PICP={picp*100:5.1f}%  mean width={2*q:5.2f}")
    # The literature-pool conformal half-width is CONSERVATIVE (literature scatter ~10
    # MPa). For the slab decision the operational sigma is the deployment error of the
    # RECOMMENDED estimator (hybrid, ambient t_eq + T_early) on the Exp#2 cored points:
    # slab_validation.csv Exp2/Cored/ambient/hybridH1 RMSE = 3.01 MPa -> 3.0.
    # Sensitivity: the core-t_eq variant gives 3.8 MPa (strike moves ~7 -> ~8 d).
    SLAB_OP_RMSE = 3.0
    sigma = SLAB_OP_RMSE
    print(f"  operational sigma for slab decision = {sigma:.2f} MPa "
          f"(from T5/T6 slab Cored hybrid RMSE)")

    # ---- quantile GB (asymmetric) for comparison ----
    full_pm = PriorModel().fit(pool)
    sh_full = full_pm.predict(pool)
    Xfull = np.column_stack([base.fillna(med).values, sh_full])
    r = y - sh_full
    qmods = {}
    for a in [0.05, 0.5, 0.95]:
        gq = GradientBoostingRegressor(loss="quantile", alpha=a, random_state=RANDOM_STATE)
        gq.fit(Xfull, r); qmods[a] = gq
    print(f"  quantile-GB 90% : mean width="
          f"{np.mean((sh_full+qmods[0.95].predict(Xfull))-(sh_full+qmods[0.05].predict(Xfull))):.2f} MPa")

    # ---- DECISION LAYER on the slab (Exp#2, fck=32) ----
    # Monotone point curve = the mix's ASTM C1074 calibrated hyperbolic (Su, k from T2
    # sanity-check = paper Table 4 refined). S(M) is monotone increasing, ->0 as M->0,
    # asymptote Su -> physically admissible BY CONSTRUCTION (no non-monotone/negative/
    # >Su predictions). The ML supplies the UNCERTAINTY band, turning the deterministic
    # maturity curve into a risk-based strike decision.
    print("\n[DECISION LAYER] slab Exp#2 (f'c=32 MPa): exceedance prob vs equivalent age")
    print("  point curve = ASTM C1074 hyperbolic Su=35.15, k=0.0191 (calibrated, R2=0.96)")
    SU_E2, K_E2 = 35.15, 0.0191
    fck = 32.0
    thr = {"KCS_14MPa": CODE_THRESHOLDS["KCS_absolute_MPa"],
           "KCS_2/3fck": CODE_THRESHOLDS["KCS_frac_fck"] * fck,
           "ACI_0.7fck": CODE_THRESHOLDS["ACI_frac_fck"] * fck}
    ages_h = np.array([12, 24, 48, 72, 120, 168, 336, 672])
    point = SU_E2 * K_E2 * ages_h / (1.0 + K_E2 * ages_h)   # monotone by construction
    rows = []
    for a, mu in zip(ages_h, point):
        rec = dict(te_h=a, te_day=round(a / 24, 2), S_pred=round(float(mu), 2))
        for name, t in thr.items():
            rec[f"P>={name}"] = round(float(1 - norm.cdf(t, loc=mu, scale=sigma)), 3)
        rows.append(rec)
    dec = pd.DataFrame(rows)
    print(dec.to_string(index=False))

    # recommended strike age: earliest te where governing criterion P>=0.95
    gov = "P>=KCS_2/3fck"     # 2/3 fck = 21.3 MPa governs over 14 MPa for fck=32
    ok = dec[dec[gov] >= 0.95]
    if len(ok):
        r0 = ok.iloc[0]
        print(f"\n  -> Recommended formwork strike (KCS, P>=0.95 that S>=2/3 f'c=21.3 MPa): "
              f"equivalent age >= {r0.te_h:.0f} h ({r0.te_day:.1f} d), S_pred={r0.S_pred} MPa")

    dec.to_csv(os.path.join(OUT_DIR, "decision_exceedance.csv"), index=False, encoding=CSV_ENC)
    print("\nWritten -> decision_exceedance.csv")


if __name__ == "__main__":
    main()
