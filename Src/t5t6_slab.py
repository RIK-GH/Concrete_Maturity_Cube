"""
t5t6_slab.py  (TASKS T5 + T6 + real-goal validation)

Leave-one-EXPERIMENT-out: train ONLY on literature, test on the slab in-situ cores
(the true out-of-domain generalisation the paper cares about for formwork removal).

L2 (ambient<->core gap):
  The paper fed AMBIENT-based maturity and underpredicted early ages (-5.37 MPa @1d,
  MAPE 27.8% @1d on Exp#2). We compute the slab maturity BOTH ways and show that
  using CORE-based equivalent age (from embedded FBG temperature) removes the
  systematic early-age underprediction. The 400mm cube quantifies the mechanism:
  core peaks ~+17 C above ambient, so ambient maturity understates early hydration.

L1 carry-over: the predictor is the fold-safe physics HYBRID (hyperbolic prior +
  residual GB), trained on the large 503 literature pool, vs the paper's pure GB.

T6 (early age): a <3d / >=3d two-regime option and per-age reporting; target
  1d & 3d MAPE < 10%.

Baseline to beat (paper, GB -> Exp#2): RMSE 3.84, MAE 3.26, R2 0.747, MAPE 14.9%.
"""
import os
import numpy as np
import pandas as pd
from config import OUT_DIR, CSV_ENC, TREF_K, EA_DEFAULT, R_GAS
from ml_utils import make_models, metrics
from physics_prior import PriorModel
import maturity as mat

MASTER = os.path.join(OUT_DIR, "master_phys.parquet")
pd.set_option("display.width", 200)

# maturity-consistent set + crossover-relevant early curing temperature.
# T_early breaks the pure-maturity assumption so the model can learn the temperature
# crossover effect (high early temperature -> lower strength at a given maturity),
# which the literature's isothermal 5/20/40 C series encode. See the ablation in main().
USE_CROSSOVER = True


BASE_FEATS = ["t_eq", "log_teq", "sqrt_teq", "WC", "C", "RH"]


def build_feats(df, te, feats):
    """Feature frame from an equivalent-age array `te` (hours) for the given feature list."""
    te = np.asarray(te, float)
    cols = {}
    cols["t_eq"] = te
    cols["log_teq"] = np.log(np.clip(te, 1e-3, None))
    cols["sqrt_teq"] = np.sqrt(np.clip(te, 0, None))
    cols["WC"] = df["WC"].values
    cols["C"] = df["C"].values
    cols["RH"] = df["RH"].values
    if "T_early" in feats:
        cols["T_early"] = df["T_early"].values
    return pd.DataFrame({k: cols[k] for k in feats}, index=df.index)


def fit_predict(train, Xtr, ytr, Xte, hybrid, sh_tr=None, sh_te=None):
    gb = make_models()["GradientBoost"]
    if hybrid:
        gb.fit(Xtr, ytr - sh_tr)
        return sh_te + gb.predict(Xte)
    gb.fit(Xtr, ytr)
    return gb.predict(Xte)


def main():
    print("=" * 78)
    print("TASKS T5+T6 - slab in-situ leave-one-experiment-out (ambient vs core)")
    print("=" * 78)
    df = pd.read_parquet(MASTER)

    # ---- training pool: large ambient literature (503) + curated 119 ----
    # Regime-match to the slab (normal-strength C20/25, w/c 0.35-0.70): training on
    # 90 MPa HSC mixes injects noise irrelevant to the slab domain.
    train = df[df.dataset.isin(["lit503", "lit119"])].copy()
    train = train[train.f_c.notna() & train.t_eq.notna() & (train.f_c > 0) &
                  (train.f_c <= 60) & train.WC.between(0.35, 0.72)].reset_index(drop=True)
    # impute C, RH medians for training rows lacking them
    for c in ["C", "RH"]:
        train[c] = train[c].fillna(train[c].median())
    # crossover feature: for isothermal literature, early curing temp = curing temp
    train["T_early"] = train["curing_C"].fillna(train.get("T_amb")).fillna(20.0)
    ytr = train["f_c"].values
    pm = PriorModel().fit(train)              # prior is feature-set independent
    sh_tr = pm.predict(train)
    print(f"train pool: {len(train)} rows, {train.mix_id.nunique()} mixes "
          f"({train.mix_family.nunique()} families)")

    # ---- slab test set (Cored = in-situ target; also SC/AD reported) ----
    slab = df[df.dataset == "slab"].copy()
    slab["exp"] = slab["mix_id"].str.replace("slab_", "")
    # ambient equivalent age from average ambient temperature (isothermal) - paper style
    slab["te_amb"] = mat.eq_age_iso(slab["curing_C"], slab["time_h"], EA_DEFAULT, TREF_K, R_GAS)
    slab["te_core"] = slab["t_eq"]           # from FBG core history (Table 3 summary)
    # crossover early-temperature: default to the paper's average ambient (isothermal proxy)
    slab["Tearly_core"] = slab["curing_C"]
    slab["Tearly_amb"] = slab["curing_C"]
    # EXACT per-age maturity + early-temperature from the raw 2-min logs (T8), when present.
    cm_path = os.path.join(OUT_DIR, "core_maturity.csv")
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path)
        used = []
        for exp in ["Exp1", "Exp2"]:
            sub = cm[cm.member == f"slab_{exp}"].set_index("age_d")
            if not len(sub):
                continue
            used.append(exp)
            for i, r in slab.iterrows():
                if r["exp"] == exp and r["age_day"] in sub.index:
                    slab.at[i, "te_core"] = float(sub.loc[r["age_day"], "teq_core"])
                    slab.at[i, "te_amb"] = float(sub.loc[r["age_day"], "teq_amb"])
                    slab.at[i, "Tearly_core"] = float(sub.loc[r["age_day"], "Tearly_core"])
                    slab.at[i, "Tearly_amb"] = float(sub.loc[r["age_day"], "Tearly_amb"])
        print(f"  [core maturity] EXACT integrated logs (T8) applied to slab {', '.join(used)}")
    for c in ["C", "RH"]:
        slab[c] = slab[c].fillna(train[c].median())

    def evaluate(feats):
        """Train on the literature pool with the given feature list; predict the slab."""
        Xtr = build_feats(train, train["t_eq"].values, feats)
        Xtr_med = Xtr.median()
        Xtr = Xtr.fillna(Xtr_med).values
        rows = []
        for exp in ["Exp1", "Exp2"]:
            for ctype in ["Cored", "AD", "SC"]:
                g = slab[(slab.exp == exp) & (slab.curing_type == ctype)]
                if len(g) == 0:
                    continue
                y = g["f_c"].values
                for mk, te, tearly in [("ambient", g["te_amb"].values, g["Tearly_amb"].values),
                                       ("core", g["te_core"].values, g["Tearly_core"].values)]:
                    gv = g.assign(T_early=tearly)
                    Xte = build_feats(gv, te, feats).fillna(Xtr_med).values
                    sh_te = pm.predict(gv.assign(t_eq=te))
                    for hyb, hname in [(False, "pureGB"), (True, "hybridH1")]:
                        pred = np.clip(fit_predict(train, Xtr, ytr, Xte, hyb,
                                                   sh_tr, sh_te if hyb else None), 0, None)
                        err = y - pred
                        e1 = err[g.age_day.values <= 1]
                        e3 = err[(g.age_day.values > 1) & (g.age_day.values <= 3)]
                        rows.append(dict(exp=exp, curing=ctype, maturity=mk, model=hname,
                                         **metrics(y, pred),
                                         err_1d=float(e1.mean()) if len(e1) else np.nan,
                                         err_3d=float(e3.mean()) if len(e3) else np.nan))
        return pd.DataFrame(rows)

    # ---- CROSSOVER ABLATION: does the early-temperature feature help? ----
    print("\n[Crossover ablation]  Exp#2 hybrid (aggregate SC/AD/Cored), train=literature")
    for label, feats in [("baseline (maturity only)", BASE_FEATS),
                         ("+ T_early (crossover)", BASE_FEATS + ["T_early"])]:
        r = evaluate(feats)
        e2 = r[(r.exp == "Exp2") & (r.model == "hybridH1")].groupby("maturity").agg(
            RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"), R2=("R2", "mean"))
        best = e2.RMSE.idxmin()
        print(f"  {label:26s}: ambient RMSE {e2.loc['ambient','RMSE']:.2f}/MAPE {e2.loc['ambient','MAPE']:.1f}% "
              f"| core RMSE {e2.loc['core','RMSE']:.2f}  -> best={best} {e2.RMSE.min():.2f}")

    # primary run uses the configured feature set
    primary_feats = BASE_FEATS + (["T_early"] if USE_CROSSOVER else [])
    res = evaluate(primary_feats)
    res.to_csv(os.path.join(OUT_DIR, "slab_validation.csv"), index=False, encoding=CSV_ENC)

    # ---- headline: Cored in-situ, the decision target ----
    print("\n[SLAB CORED in-situ prediction]  (train=literature only)")
    cored = res[res.curing == "Cored"]
    print(cored[["exp", "maturity", "model", "RMSE", "MAE", "R2", "MAPE", "err_1d", "err_3d"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\n[Exp#2 vs paper baseline]  paper GB: RMSE 3.84, MAE 3.26, R2 0.747, MAPE 14.9%")
    e2 = res[(res.exp == "Exp2")].groupby(["maturity", "model"]).agg(
        RMSE=("RMSE", "mean"), MAPE=("MAPE", "mean"), R2=("R2", "mean")).reset_index()
    print(e2.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ---- L2 ambient->core early-age improvement (Cored + AD, where early ages exist) ----
    print("\n[L2: early-age error, ambient vs core maturity]  (mean signed err, MPa; neg=underpredict)")
    ea = res[res.model == "hybridH1"].groupby("maturity").agg(
        err_1d=("err_1d", "mean"), err_3d=("err_3d", "mean"),
        MAPE=("MAPE", "mean")).reset_index()
    print(ea.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # ---- per-age detail: Exp2 Cored, measured vs predicted (ambient & core) ----
    print("\n[Exp#2 Cored: per-age measured vs predicted]  (hybridH1)")
    g = slab[(slab.exp == "Exp2") & (slab.curing_type == "Cored")].sort_values("age_day")
    ytrue = g["f_c"].values
    Xtr_p = build_feats(train, train["t_eq"].values, primary_feats)
    Xtr_med = Xtr_p.median(); Xtr_p = Xtr_p.fillna(Xtr_med).values
    row = []
    for mat_kind, te, tearly in [("ambient", g["te_amb"].values, g["Tearly_amb"].values),
                                 ("core", g["te_core"].values, g["Tearly_core"].values)]:
        gv = g.assign(T_early=tearly)
        Xte = build_feats(gv, te, primary_feats).fillna(Xtr_med).values
        sh_te = pm.predict(gv.assign(t_eq=te))
        pred = np.clip(fit_predict(train, Xtr_p, ytr, Xte, True, sh_tr, sh_te), 0, None)
        for a, yt, yp in zip(g.age_day.values, ytrue, pred):
            row.append(dict(age_d=a, maturity=mat_kind, measured=yt, predicted=round(yp, 2),
                            err=round(yt - yp, 2), ape=round(abs(yt - yp) / yt * 100, 1)))
    pa = pd.DataFrame(row).pivot_table(index="age_d", columns="maturity",
                                       values=["measured", "predicted", "ape"])
    print(pa.to_string(float_format=lambda v: f"{v:.2f}"))

    # ---- cube bridge: ambient vs core maturity offset from the REAL 2-min logs (T8) ----
    print("\n[Cube bridge: ambient vs core equivalent-age offset from raw logs]")
    cm_path = os.path.join(OUT_DIR, "core_maturity.csv")
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path)
        for member, g in cm.groupby("member"):
            g = g[g.age_d.isin([1, 3, 7])]
            ratios = ", ".join(f"{int(r.age_d)}d={r.ratio_teq:.2f}x" for _, r in g.iterrows())
            print(f"   {member:>10}: core/ambient t_eq  {ratios}")
        print("   (measured early-age offset ~1.4-1.6x; earlier peak-isothermal estimate 2.34x was high)")

    print("\nWritten -> slab_validation.csv")


if __name__ == "__main__":
    main()
