"""
make_figures.py  - the four required figures (-> ../figures/).
  (i)   unified maturity-strength scatter + per-mix hyperbolic curves
  (ii)  row-LOO vs Leave-One-Mix-Out performance (the leakage collapse) + big-pool
  (iii) slab per-age error, ambient vs core maturity, pure GB vs hybrid
  (iv)  ASTM C1074 strike curve with uncertainty band + code exceedance probabilities
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUT_DIR, FIG_DIR, ROOT_DIR

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
C = {"lit": "#4C78A8", "slab": "#E45756", "cube": "#54A24B", "hyp": "#333333",
     "amb": "#F58518", "core": "#72B7B2", "pure": "#B279A2", "hybrid": "#4C78A8"}


def fig1_scatter():
    df = pd.read_parquet(os.path.join(OUT_DIR, "master_phys.parquet"))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for ds, c, lab in [("lit503", C["lit"], "literature (503)"),
                       ("lit55", "#9ecae1", "paper 55"),
                       ("cube", C["cube"], "400mm cube"),
                       ("slab", C["slab"], "slab in-situ")]:
        g = df[df.dataset == ds]
        ax.scatter(g.t_eq / 24, g.f_c, s=16, alpha=0.5, c=c, label=lab, edgecolors="none")
    # slab hyperbolic curves
    M = np.linspace(1, 1000, 300)
    for su, k, lab, c in [(22.85, 0.0099, "Exp1 Su=22.85", C["slab"]),
                          (35.15, 0.0191, "Exp2 Su=35.15", "#7a1f1f")]:
        ax.plot(M / 24, su * k * M / (1 + k * M), "--", c=c, lw=1.6, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("equivalent age  $t_{eq}$  (days, log)")
    ax.set_ylabel("compressive strength (MPa)")
    ax.set_title("(i) Unified maturity–strength dataset + ASTM C1074 hyperbolic")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig1_maturity_scatter.png")); plt.close(fig)


def fig2_cv():
    cv = pd.read_csv(os.path.join(OUT_DIR, "t3t4_cv_results.csv"))
    big = pd.read_csv(os.path.join(OUT_DIR, "t3t4_bigpool_results.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # panel A: 55-point row-LOO vs mix-CV RMSE
    piv = cv.pivot_table(index="model", columns="cv", values="RMSE")
    order = ["GradientBoost", "RandomForest", "Ridge_a0.1", "Ensemble(top3)",
             "H1_residual", "H3_GB+physfeat", "H2_monotonicXGB"]
    piv = piv.reindex([m for m in order if m in piv.index])
    x = np.arange(len(piv)); w = 0.38
    axes[0].bar(x - w/2, piv.iloc[:, 0], w, label=piv.columns[0], color="#9ecae1")
    axes[0].bar(x + w/2, piv.iloc[:, 1], w, label=piv.columns[1], color=C["slab"])
    axes[0].axhline(1.82, ls=":", c="gray", label="paper Ensemble 1.82")
    axes[0].set_xticks(x); axes[0].set_xticklabels(piv.index, rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE (MPa)")
    axes[0].set_title("(ii-a) 55 pts: row-LOO leaks vs mix-CV honest")
    axes[0].legend(fontsize=8)
    # panel B: big pool pure vs hybrid
    big = big.set_index("model").reindex(
        ["GradientBoost", "RandomForest", "Ridge_a0.1", "H1_residual",
         "H3_GB+physfeat", "H2_monotonicXGB"])
    cols = ["#B279A2" if "H" not in m[:1] or m.startswith("H")==False else C["hybrid"] for m in big.index]
    colors = [C["hybrid"] if m.startswith("H") else C["pure"] for m in big.index]
    axes[1].bar(np.arange(len(big)), big.RMSE, color=colors)
    axes[1].set_xticks(np.arange(len(big))); axes[1].set_xticklabels(big.index, rotation=40, ha="right", fontsize=8)
    axes[1].set_ylabel("RMSE (MPa)")
    axes[1].set_title("(ii-b) 503 pool, leave-mix-family-out\n(purple=pure ML, blue=physics-hybrid)")
    fig.savefig(os.path.join(FIG_DIR, "fig2_cv_leakage.png")); plt.close(fig)


def fig3_slab():
    sv = pd.read_csv(os.path.join(OUT_DIR, "slab_validation.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # panel A: Exp2 aggregate RMSE by model x maturity
    e2 = sv[sv.exp == "Exp2"].groupby(["maturity", "model"]).RMSE.mean().unstack()
    e2 = e2.reindex(index=["ambient", "core"])
    x = np.arange(len(e2)); w = 0.38
    axes[0].bar(x - w/2, e2["pureGB"], w, label="pure GB", color=C["pure"])
    axes[0].bar(x + w/2, e2["hybridH1"], w, label="hybrid H1", color=C["hybrid"])
    axes[0].axhline(3.84, ls=":", c="k", label="paper GB 3.84")
    axes[0].set_xticks(x); axes[0].set_xticklabels(e2.index)
    axes[0].set_xlabel("slab maturity source"); axes[0].set_ylabel("RMSE (MPa)")
    axes[0].set_title("(iii-a) slab Exp#2 prediction (train=literature)")
    axes[0].legend(fontsize=8)
    # panel B: early-age signed error
    ea = sv[sv.model == "hybridH1"].groupby("maturity")[["err_1d", "err_3d"]].mean()
    ea = ea.reindex(["ambient", "core"])
    x = np.arange(2); w = 0.35
    axes[1].bar(x - w/2, ea["err_1d"], w, label="err @1d", color=C["amb"])
    axes[1].bar(x + w/2, ea["err_3d"], w, label="err @3d", color=C["core"])
    axes[1].axhline(0, c="k", lw=0.8)
    axes[1].axhline(-5.37, ls=":", c=C["slab"], label="paper -5.37 @1d")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ea.index)
    axes[1].set_ylabel("mean signed error (MPa)")
    axes[1].set_title("(iii-b) early-age bias removed\n(measured - predicted)")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig3_slab_ambient_core.png")); plt.close(fig)


SIGMA_OP = 3.0   # deployment RMSE of the recommended hybrid on Exp#2 cored points (T5/T6)


def fig4_decision():
    dec = pd.read_csv(os.path.join(OUT_DIR, "decision_exceedance.csv"))
    sigma = SIGMA_OP
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # panel A: strength curve + band
    axes[0].plot(dec.te_day, dec.S_pred, "-o", c=C["hybrid"], label="ASTM C1074 curve")
    axes[0].fill_between(dec.te_day, dec.S_pred - 1.645*sigma, dec.S_pred + 1.645*sigma,
                         alpha=0.2, color=C["hybrid"], label="90% band")
    for t, lab, c in [(14, "14 MPa (KCS)", "#888"), (21.3, "2/3 f'c", C["amb"]),
                      (22.4, "0.7 f'c (ACI)", C["slab"])]:
        axes[0].axhline(t, ls="--", c=c, lw=1, label=lab)
    axes[0].set_xscale("log"); axes[0].set_xlabel("equivalent age (days, log)")
    axes[0].set_ylabel("strength (MPa)")
    axes[0].set_title("(iv-a) slab Exp#2 strike curve + uncertainty")
    axes[0].legend(fontsize=7, loc="lower right")
    # panel B: exceedance probabilities
    for col, c, lab in [("P>=KCS_14MPa", "#888", "P(S>=14)"),
                        ("P>=KCS_2/3fck", C["amb"], "P(S>=2/3 f'c)"),
                        ("P>=ACI_0.7fck", C["slab"], "P(S>=0.7 f'c)")]:
        axes[1].plot(dec.te_day, dec[col], "-o", c=c, label=lab)
    axes[1].axhline(0.95, ls=":", c="k", label="0.95 target")
    axes[1].axvline(7, ls="--", c="green", alpha=0.6, label="strike @7d")
    axes[1].set_xscale("log"); axes[1].set_xlabel("equivalent age (days, log)")
    axes[1].set_ylabel("exceedance probability")
    axes[1].set_title("(iv-b) risk-based formwork-removal criterion")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig4_decision.png")); plt.close(fig)


def fig5_core_logs():
    """Raw core-temperature histories + maturity-integration validation vs paper."""
    import os
    cm = pd.read_csv(os.path.join(OUT_DIR, "core_maturity.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # panel A: raw core temperature histories - both real slab experiments
    try:
        e1 = pd.read_excel(os.path.join(ROOT_DIR, "SLAB_Exp_1.xlsx"), sheet_name=0)
        d1 = pd.to_numeric(e1["days"], errors="coerce")
        c1 = e1[[c for c in e1.columns if "FBG" in str(c)]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        axes[0].plot(d1, c1, c="#C44E00", lw=1.4, label="Exp#1 core (summer)")
        axes[0].plot(d1, pd.to_numeric(e1["Ambient Temp"], errors="coerce"),
                     c="#C44E00", lw=0.9, ls=":", alpha=0.6, label="Exp#1 ambient")
    except Exception as e:
        print("Exp1 log plot skipped:", e)
    try:
        sl = pd.read_csv(os.path.join(ROOT_DIR, "SLAB_Exp_2", "A", "FBGA_IoT.csv"), encoding="utf-8-sig")
        d = pd.to_numeric(sl["days"], errors="coerce")
        core = sl[[c for c in sl.columns if str(c).strip().upper().startswith("FBG")]].apply(
            pd.to_numeric, errors="coerce").mean(axis=1)
        amb = pd.to_numeric(sl[[c for c in sl.columns if "Temperature" in str(c)][-1]], errors="coerce")
        axes[0].plot(d, core, c=C["slab"], lw=1.4, label="Exp#2 core (autumn)")
        axes[0].plot(d, amb, c=C["slab"], lw=0.9, ls=":", alpha=0.6, label="Exp#2 ambient")
    except Exception as e:
        print("Exp2 log plot skipped:", e)
    try:
        cu = pd.read_excel(os.path.join(ROOT_DIR, "CUBE_TC-Mockup.xlsx"), sheet_name=0)
        dt = pd.to_datetime(cu["datetime"], errors="coerce")
        th = (dt - dt.iloc[0]).dt.total_seconds() / 86400
        for cube, wc, c in [("A", 0.555, C["cube"]), ("C", 0.60, C["amb"])]:
            core = cu[[f"TC-{cube}-{i}" for i in (1, 2, 3)]].apply(
                pd.to_numeric, errors="coerce").mean(axis=1)
            axes[0].plot(th, core, c=c, lw=1.2, alpha=0.8, label=f"cube w/c{wc} core")
    except Exception as e:
        print("cube log plot skipped:", e)
    axes[0].set_xlim(0, 7); axes[0].set_xlabel("age (days)")
    axes[0].set_ylabel("temperature (°C)")
    axes[0].set_title("(v-a) raw core-temperature logs (real, 2-min)")
    axes[0].legend(fontsize=7)
    # panel B: slab core t_eq vs paper Table 3, BOTH experiments
    paper = {"slab_Exp1": {3: 123.83, 7: 266.22, 28: 947.59},
             "slab_Exp2": {1: 28.86, 3: 80.19, 7: 163.63, 14: 310.15, 28: 572.11}}
    for mem, c, lab in [("slab_Exp1", "#C44E00", "Exp#1"), ("slab_Exp2", C["slab"], "Exp#2")]:
        sl = cm[cm.member == mem]
        axes[1].plot(sl.age_d, sl.teq_core, "-o", c=c, label=f"{lab} core t_eq (ours)")
        pk = paper[mem]
        axes[1].plot(list(pk.keys()), list(pk.values()), "s", c="k", mfc="none", ms=8)
    axes[1].plot([], [], "sk", mfc="none", label="paper Table 3")
    axes[1].set_xlabel("age (days)"); axes[1].set_ylabel("equivalent age $t_{eq}$ (h)")
    axes[1].set_title("(v-b) core maturity validated vs paper\n(Exp#1 947 h vs Exp#2 572 h @28 d)")
    axes[1].legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "fig5_core_logs.png")); plt.close(fig)


def fig6_calibration():
    """In-place ASTM C1074 calibration: literature-only vs calibrated, both experiments."""
    import os
    cal = pd.read_csv(os.path.join(OUT_DIR, "calibration_results.csv"))
    cm = pd.read_csv(os.path.join(OUT_DIR, "core_maturity.csv"))
    CORED = {"Exp1": {3: 13.00, 7: 15.90, 28: 20.90},
             "Exp2": {3: 21.30, 7: 27.20, 14: 28.73, 28: 33.00}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, exp in zip(axes, ["Exp1", "Exp2"]):
        sub = cm[cm.member == f"slab_{exp}"].set_index("age_d")
        ages = sorted(CORED[exp])
        te = np.array([float(sub.loc[a, "teq_core"]) for a in ages])
        S = np.array([CORED[exp][a] for a in ages])
        ax.scatter(te, S, c="k", zorder=5, label="measured Cored")
        M = np.linspace(10, te.max() * 1.05, 200)
        for mode, c, ls in [("literature-only", C["pure"], "--"), ("calib<= 3d", C["hybrid"], "-")]:
            r = cal[(cal.exp == exp) & (cal["mode"] == mode)]
            if len(r):
                su, k = r.Su.iloc[0], r.k.iloc[0]
                rmse = r.RMSE_later.iloc[0]
                ax.plot(M, su * k * M / (1 + k * M), ls, c=c,
                        label=f"{mode} (RMSE {rmse})")
        ax.set_xlabel("core equivalent age (h)"); ax.set_ylabel("strength (MPa)")
        ax.set_title(f"(vi) {exp}: in-place calibration"
                     + ("  W/C 0.64 (hard)" if exp == "Exp1" else "  W/C 0.44"))
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig6_calibration.png")); plt.close(fig)


def fig_decision_cube():
    """Fig 15: the decision layer applied to the cube (W/C 0.50, assumed f'c=24 MPa),
    with the slab Exp#2 curve for comparison. Point curves are the members' ASTM C1074
    hyperbolic fits; sigma = SIGMA_OP."""
    from scipy.stats import norm
    hf = pd.read_csv(os.path.join(OUT_DIR, "hyperbolic_fits.csv"))
    su_c, k_c = hf.loc[hf.mix_id == "cube_wc500", ["Su", "k"]].iloc[0]
    SU_E2, K_E2 = 35.15, 0.0191                     # slab Exp#2 cored fit (as in T7)
    fck_cube, fck_slab = 24.0, 32.0
    thr_cube = max(14.0, 2 * fck_cube / 3)          # KCS: max(14, 2/3 f'c) = 16
    thr_slab = 2 * fck_slab / 3                     # 21.3
    te = np.linspace(6, 900, 500)
    S_c = su_c * k_c * te / (1 + k_c * te)
    S_s = SU_E2 * K_E2 * te / (1 + K_E2 * te)
    P_c = 1 - norm.cdf(thr_cube, loc=S_c, scale=SIGMA_OP)
    P_s = 1 - norm.cdf(thr_slab, loc=S_s, scale=SIGMA_OP)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(te / 24, S_c, "-", c=C["cube"], label="cube ASTM C1074 curve (W/C 0.50)")
    axes[0].fill_between(te / 24, S_c - 1.645 * SIGMA_OP, S_c + 1.645 * SIGMA_OP,
                         alpha=0.2, color=C["cube"], label="90% band")
    for t, lab, c in [(14, "14 MPa (KCS floor)", "#888"),
                      (thr_cube, "2/3 f'c = 16 MPa", C["amb"])]:
        axes[0].axhline(t, ls="--", c=c, lw=1, label=lab)
    axes[0].set_xscale("log"); axes[0].set_xlabel("equivalent age (days, log)")
    axes[0].set_ylabel("strength (MPa)")
    axes[0].set_title("(a) cube strike curve + uncertainty")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].plot(te / 24, P_c, "-", c=C["cube"], label="cube: P(S >= 2/3 f'c = 16)")
    axes[1].plot(te / 24, P_s, "--", c=C["slab"], label="slab Exp#2: P(S >= 2/3 f'c = 21.3)")
    axes[1].axhline(0.95, ls=":", c="k", label="0.95 target")
    for arr, c in [(P_c, C["cube"]), (P_s, C["slab"])]:
        i = np.argmax(arr >= 0.95)
        axes[1].axvline(te[i] / 24, ls=":", c=c, alpha=0.6)
    axes[1].set_xscale("log"); axes[1].set_xlabel("equivalent age (days, log)")
    axes[1].set_ylabel("exceedance probability")
    axes[1].set_title("(b) code-referenced exceedance, cube vs slab")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "fig_decision_cube.png")); plt.close(fig)
    for name, arr in [("cube", P_c), ("slab", P_s)]:
        i = np.argmax(arr >= 0.95)
        print(f"  [deccube] {name}: P>=0.95 at te = {te[i]:.0f} h ({te[i]/24:.1f} d)")


if __name__ == "__main__":
    fig1_scatter(); fig2_cv(); fig3_slab(); fig4_decision(); fig5_core_logs()
    fig6_calibration(); fig_decision_cube()
    print("figures written to", FIG_DIR)
    for f in sorted(os.listdir(FIG_DIR)):
        print("  ", f)
