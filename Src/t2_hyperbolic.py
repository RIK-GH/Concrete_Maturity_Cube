"""
t2_hyperbolic.py  (TASK T2)  -  Physics-based mechanistic baseline (prior).

ASTM C1074 hyperbolic strength-maturity law, calibrated per mix on equivalent age:
        S(M) = Su * k * M / (1 + k * M)          (M = t_eq in hours)

For each mix with >= MIN_PTS strength-maturity points we fit (Su, k) by the
paper's iterative least-squares (reproduced with scipy). This S_hyp becomes the
PHYSICS PRIOR consumed by the hybrid ML (T3, residual learning) and enforces the
correct asymptote (Su), monotonicity, and S(M->0)->0.

Rows whose mix cannot be fit individually (sparse families in the 503 pool) get a
POOLED fallback prior: Su tied to the mix's own max observed strength, k from a
global median. Every row therefore carries a finite, monotone, physical S_hyp.

Sanity check: slab Cored fits must reproduce paper Table 4
        Exp1: Su=22.85, k=0.0099 (R2 0.979) ; Exp2: Su=32.86, k=0.0191 (R2 0.961)
"""
import os
import numpy as np
import pandas as pd
from config import OUT_DIR, CSV_ENC
from maturity import _hyperbolic_age, fit_rate_constant

MIN_PTS = 3          # minimum (M, S) points to fit a per-mix hyperbolic (2 params)
MIN_SPAN = 2.5       # require max(M)/min(M) >= this so Su (asymptote) is identifiable
MASTER = os.path.join(OUT_DIR, "master.parquet")


def hyperbolic_M(M, su, k):
    return su * k * M / (1.0 + k * M)


def fit_mix(M, S):
    """Fit S = Su k M/(1+kM). Returns Su, k, r2, rmse, n."""
    from scipy.optimize import curve_fit
    M = np.asarray(M, float); S = np.asarray(S, float)
    ok = np.isfinite(M) & np.isfinite(S) & (M > 0)
    M, S = M[ok], S[ok]
    n = len(M)
    if n < MIN_PTS or (M.max() / max(M.min(), 1e-6)) < MIN_SPAN:
        return None
    try:
        popt, _ = curve_fit(hyperbolic_M, M, S, p0=[S.max() * 1.25, 0.01],
                            bounds=([S.max() * 0.9, 1e-5], [S.max() * 2.0 + 20, 5.0]),
                            maxfev=20000)
    except Exception:
        return None
    su, k = popt
    pred = hyperbolic_M(M, su, k)
    rmse = float(np.sqrt(np.mean((S - pred) ** 2)))
    ss_res = np.sum((S - pred) ** 2); ss_tot = np.sum((S - S.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return dict(Su=float(su), k=float(k), r2=r2, rmse=rmse, n=n)


def main():
    print("=" * 78)
    print("TASK T2 - ASTM C1074 hyperbolic mechanistic baseline (physics prior)")
    print("=" * 78)
    df = pd.read_parquet(MASTER)
    df["S_hyp"] = np.nan
    df["Su_mix"] = np.nan
    df["k_mix"] = np.nan

    fit_records = []
    global_ks = []

    # ---- per-mix fits (use t_eq as maturity M, hours) ----
    for (dataset, mid), g in df.groupby(["dataset", "mix_id"]):
        # for slab use ONLY the curing type that defines the mix curve (Cored) for the
        # sanity check, but fit each curing_type separately where present
        res = fit_mix(g["t_eq"], g["f_c"])
        if res is not None:
            df.loc[g.index, "S_hyp"] = hyperbolic_M(g["t_eq"].values, res["Su"], res["k"])
            df.loc[g.index, "Su_mix"] = res["Su"]
            df.loc[g.index, "k_mix"] = res["k"]
            global_ks.append(res["k"])
            fit_records.append(dict(dataset=dataset, mix_id=mid, **res))

    fits = pd.DataFrame(fit_records)
    k_med = float(np.median(global_ks)) if global_ks else 0.02
    print(f"\nPer-mix hyperbolic fits: {len(fits)} mixes fitted (>= {MIN_PTS} pts). "
          f"global median k = {k_med:.4f}/h")

    # ---- pooled Su(w/c) fallback for un-fitted (sparse) rows ----
    # Physical anchor: ultimate strength rises as w/c falls (Abrams/Bolomey).
    # Regress the well-identified per-mix Su on 1/(w/c) and predict for sparse mixes,
    # rather than using each mix's max observed strength (which is far below Su when
    # only early-age points exist -> the earlier RMSE=17 artefact).
    fitpool = fits.merge(df[["mix_id", "WC"]].drop_duplicates(), on="mix_id", how="left")
    fitpool = fitpool[fitpool.WC.notna() & (fitpool.Su < 120)]
    if len(fitpool) >= 3:
        a, b = np.polyfit(1.0 / fitpool.WC, fitpool.Su, 1)   # Su ~ a*(1/wc)+b
    else:
        a, b = 0.0, float(np.median(fits.Su)) if len(fits) else 45.0
    print(f"Pooled Su(w/c) prior:  Su_hat = {a:.2f}*(1/wc) + {b:.2f}")

    unfit = df["S_hyp"].isna()
    for mid, g in df[unfit].groupby("mix_id"):
        wc = g["WC"].dropna()
        if len(wc):
            su_hat = float(np.clip(a / wc.iloc[0] + b, g["f_c"].max() * 1.02, 130))
        else:
            su_hat = float(g["f_c"].max()) * 1.15
        df.loc[g.index, "S_hyp"] = hyperbolic_M(g["t_eq"].values, su_hat, k_med)
        df.loc[g.index, "Su_mix"] = su_hat
        df.loc[g.index, "k_mix"] = k_med
    print(f"Pooled-fallback prior applied to {int(unfit.sum())} rows "
          f"({df['mix_id'][unfit].nunique()} sparse mixes).")

    df["S_hyp"] = df["S_hyp"].clip(lower=0)
    df["resid_phys"] = df["f_c"] - df["S_hyp"]     # target for residual learning (T3)

    # ---- SLAB sanity check vs paper Table 4 (Cored) ----
    print("\n[Sanity check] slab Cored hyperbolic vs paper Table 4")
    slab_cored = df[(df.dataset == "slab") & (df.curing_type == "Cored")]
    for exp in ["slab_Exp1", "slab_Exp2"]:
        g = slab_cored[slab_cored.mix_id == exp]
        r = fit_mix(g["t_eq"], g["f_c"])
        if r:
            paper = {"slab_Exp1": (22.85, 0.0099, 0.979),
                     "slab_Exp2": (32.86, 0.0191, 0.961)}[exp]
            print(f"   {exp}: fit Su={r['Su']:.2f} k={r['k']:.4f} R2={r['r2']:.3f}  "
                  f"| paper Su={paper[0]} k={paper[1]} R2={paper[2]}")

    # ---- report physics prior quality per dataset ----
    print("\n[Physics prior fit quality]  (per-mix fits only)")
    if len(fits):
        rep = fits.groupby("dataset").agg(
            mixes=("mix_id", "size"), mean_r2=("r2", "mean"),
            mean_rmse=("rmse", "mean"), Su_min=("Su", "min"), Su_max=("Su", "max"))
        print(rep.to_string())

    # baseline error of the pure physics prior (before any ML correction)
    print("\n[Pure physics baseline error]  S_hyp vs measured, by dataset")
    for ds, g in df.groupby("dataset"):
        rmse = np.sqrt(np.mean(g["resid_phys"] ** 2))
        mae = np.mean(np.abs(g["resid_phys"]))
        print(f"   {ds:8s} n={len(g):4d}  RMSE={rmse:5.2f}  MAE={mae:5.2f} MPa")

    df.to_parquet(os.path.join(OUT_DIR, "master_phys.parquet"), index=False)
    fits.to_csv(os.path.join(OUT_DIR, "hyperbolic_fits.csv"), index=False, encoding=CSV_ENC)
    print(f"\nWritten -> master_phys.parquet (+ S_hyp, resid_phys), hyperbolic_fits.csv")


if __name__ == "__main__":
    main()
