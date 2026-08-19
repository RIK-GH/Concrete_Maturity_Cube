"""
vs_sweep_postproc.py  --  Volume-to-surface (least-dimension) sweep post-processing.

Turns the ABAQUS core-temperature output of several geometrically-similar members
(different least dimension L) into the curve the reviewer asked for:
    predicted in-place-to-standard behaviour  vs  member size.

Rigorous part  : core EQUIVALENT AGE (ASTM C1074 Arrhenius) from the FE core-temp history.
Optional part  : convert to strength via the calibrated hyperbolic S_hyp (state moisture caveat).

USAGE
-----
1. Run the SAME calibrated HETVAL/UMAT model (one mix, e.g. W/C 0.50) on cubes of
   L = 100, 200, 300, 400, 600, 800 mm  (identical material, BC, ambient, DURATION).
   Keep element SIZE ~constant when remeshing so the fixed skin depth (~62 mm) stays resolved.
2. Report the CENTRE-node temperature history of each run to a .rpt (time[s], T[C]).
3. Fill FILES below with {L_mm: rpt_path} and run:  python vs_sweep_postproc.py
"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- constants (match the maturity convention used throughout the paper) ----
Ea, R, Tref_K = 38300.0, 8.314, 293.15          # J/mol, J/mol/K, 20 C
# (Su, k) refit EXCLUDING the sensor-bearing 28-day companion specimens (keep 1-14 d + cores),
# per the FBG-compaction disclosure, with the core equivalent age INTEGRATED from the real
# 2-min thermocouple logs (core_maturity.csv; the earlier constant-peak placeholder is gone).
HYP = {"0.50":  (27.43, 0.02120),
       "0.555": (30.59, 0.03361),
       "0.60":  (23.49, 0.01207)}

# ---- least dimension (mm) -> ABAQUS core-temp .rpt for the chosen mix ----
# These are produced by 2026_Vs_sweep/extract_core_temp.py (2-col: time_s  T_center) -> CORE_COL=1.
MIX = "0.50"
from config import ABAQUS_DIR, FIG_DIR
SWEEP = os.path.join(ABAQUS_DIR, "2026_Vs_sweep")
FILES = {
    100: SWEEP + r"/sweep_L100_core.rpt",
    200: SWEEP + r"/sweep_L200_core.rpt",
    300: SWEEP + r"/sweep_L300_core.rpt",
    400: SWEEP + r"/sweep_L400_core.rpt",   # ACTUAL specimen size -> validation anchor
    600: SWEEP + r"/sweep_L600_core.rpt",
    # 800: regularised HETVAL (smoothstep replaces the discontinuous 0.8 rate step at 52 C
    #      that stalled the original solve; identical below 52 C, verified at L400 to 0.01 C),
    #      heat-transfer-only solve (temperature field matches the coupled one; L400 check).
    800: SWEEP + r"/sweep_L800_htr_core.rpt",
    # 450 excluded: base-mesh artifact (specimen is 400 mm)
}
CORE_COL = 1           # sweep files: col0=time_s, col1=T_center. (old temp_15_b.rpt used col2.)
OUT = os.path.join(FIG_DIR, "fig_vs_sweep.png")

def parse_temp(fn, col):
    t, T = [], []
    for ln in open(fn, encoding="latin-1"):
        tk = ln.split()
        if len(tk) > col:
            try:
                t.append(float(tk[0])); T.append(float(tk[col]))
            except ValueError:
                pass
    return np.asarray(t), np.asarray(T)

def equiv_age_h(t_s, T_C):
    """Arrhenius equivalent age (hours) at Tref, trapezoidal."""
    xi = np.exp(-Ea / R * (1.0 / (T_C + 273.15) - 1.0 / Tref_K))
    te = np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(t_s))])
    return te / 3600.0

def S_hyp(M, Su, k):
    return Su * k * M / (1.0 + k * M)

# ---- Temperature-dependent ULTIMATE strength (crossover), from the paper's own fit:
#      Su = 30 / 27 / 22 MPa at curing temp 5 / 20 / 40 C.  Localises the high-T strength
#      penalty on Su (rate k unchanged). A hot core matures faster (higher M) but tops out
#      LOWER. This is what reconciles the sweep with the measured cube: without it the strength
#      ratio is systematically overstated (Reviewer Major-1).  T_early = mean core temp, 1st 72 h.
SU_T = ([5.0, 20.0, 40.0], [30.0, 27.0, 22.0])
def su_shape(Tc):
    return float(np.interp(Tc, SU_T[0], SU_T[1])) / 27.0   # fractional Su(T)/Su(20), clamped
# anchor the crossover shape to THIS mix's own water-cured ultimate strength (works for every mix,
# incl. 0.60 whose Su_std=22.2 differs from the 0.50-based crossover). Ratio is anchor-invariant.
SU_STD = HYP[MIX][0]
def su_core_of_T(Tc):
    return SU_STD * su_shape(Tc)

def mean_early_T(t_s, T_C, hours=72.0):
    m = t_s <= hours * 3600.0
    return float(T_C[m].mean()) if m.any() else float(T_C[0])

# read the in-place/standard strength ratio at these real (wall-clock) ages.
# 2 d = ~the 14 MPa floor; 7 d = the paper's recommended striking age (Fig. decision).
STRIKE_H = [48.0, 168.0]      # 2 d, 7 d
LATE_H   = 24.0 * 14.0        # ~end-of-run late-age proxy (runs span 14.9 d)

def core_te_at(t_s, te_h, tstar_h):
    age_h = t_s / 3600.0
    if tstar_h > age_h[-1]:
        return None
    return float(np.interp(tstar_h, age_h, te_h))

def main():
    _, k = HYP[MIX]
    rows = []
    for L, fn in sorted(FILES.items()):
        if not os.path.exists(fn):
            print(f"  [skip] L={L}: file not found -> {fn}"); continue
        t, T = parse_temp(fn, CORE_COL)
        te = equiv_age_h(t, T)
        age_h = t[-1] / 3600.0
        Tearly = mean_early_T(t, T)
        Su_core = su_core_of_T(Tearly)               # hot core -> lower ultimate strength
        r = dict(L=L, coreDepth=L/2, peakT=T.max(), age_d=age_h/24, Tearly=Tearly,
                 Su_core=Su_core, mat_ratio=te[-1]/age_h)
        # strength ratio: core uses Su_core & its accelerated maturity; std uses SU_STD & isothermal age
        for tag, h in list(zip(("2d", "7d"), STRIKE_H)) + [("late", min(LATE_H, age_h))]:
            cte = core_te_at(t, te, h)
            r[f"S_{tag}"] = (S_hyp(cte, Su_core, k) / S_hyp(h, SU_STD, k)) if cte is not None else None
        rows.append(r)
    if not rows:
        print("No data."); return
    print(f"\n  Mix W/C {MIX}  (k={k}, Su_std={SU_STD:.1f}, Su(T) crossover 30/27/22 @ 5/20/40C)")
    print(f"  {'L(mm)':>6} {'core':>5} {'peakT':>6} {'Tearly':>6} {'Su_core':>7} {'mat_r':>6} "
          f"{'S_2d':>6} {'S_7d':>6} {'S_late':>7}")
    for r in rows:
        f2 = lambda v: (f"{v:.2f}" if v is not None else "--")
        print(f"  {r['L']:>6} {r['coreDepth']:>5.0f} {r['peakT']:>6.1f} {r['Tearly']:>6.1f} "
              f"{r['Su_core']:>7.1f} {r['mat_ratio']:>6.2f} "
              f"{f2(r['S_2d']):>6} {f2(r['S_7d']):>6} {f2(r['S_late']):>7}")
    # persist the table so SI Table S5 and manuscript Table 7 are reproducible, not hand-typed
    import pandas as pd
    from config import OUT_DIR, CSV_ENC
    pd.DataFrame([dict(mix_wc=float(MIX), L_mm=r["L"], core_depth_mm=r["coreDepth"],
                       peak_T_C=round(r["peakT"], 1), T_early_C=round(r["Tearly"], 1),
                       maturity_ratio=round(r["mat_ratio"], 3),
                       Su_core_MPa=round(r["Su_core"], 1),
                       strength_ratio_2d=None if r["S_2d"] is None else round(r["S_2d"], 3),
                       strength_ratio_7d=None if r["S_7d"] is None else round(r["S_7d"], 3),
                       strength_ratio_late14d=None if r["S_late"] is None else round(r["S_late"], 3))
                  for r in rows]).to_csv(
        os.path.join(OUT_DIR, f"vs_sweep_{MIX.replace('.','')}.csv"), index=False, encoding=CSV_ENC)

    anchor = [r["S_late"] for r in rows if r["L"] == 400]
    anchor_s = f"{anchor[0]:.2f}" if anchor else "n/a"
    print(f"\n  [consistency] measured 400 mm cube W/C 0.50 core/standard @28 d = 0.82-0.99 ; "
          f"model S_late@400 = {anchor_s}")
    if len(rows) >= 2:
        L = [r["L"] for r in rows]
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        ax.axhline(1.0, color="#6b7078", lw=0.8, ls="--")
        ax.plot(L, [r["S_2d"] for r in rows], "o-",  color="#b56812", label="strength ratio @ 2 d (early / floor)")
        ax.plot(L, [r["S_7d"] for r in rows], "^-",  color="#c99a3b", label="strength ratio @ 7 d (striking)")
        ax.plot(L, [r["S_late"] for r in rows], "v--", color="#7a4b2b", label="strength ratio @ ~14 d (late)")
        # measured 400 mm cube 28-d band (consistency check)
        ax.fill_between([375, 425], 0.82, 0.99, color="#3d6b7e", alpha=0.18, zorder=0)
        ax.plot([400], [0.905], "s", color="#3d6b7e", ms=6, label="measured 400 mm cube @28 d (0.82-0.99)")
        ax.set_xlabel("member least dimension L (mm)")
        ax.set_ylabel("in-place / standard strength ratio")
        ax.set_title(f"Age- and size-dependent member effect, W/C {MIX}\n(calibrated FE, temperature-dependent $S_u$)",
                     fontsize=9.5, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        plt.tight_layout(); plt.savefig(OUT, dpi=300, facecolor="white")
        print(f"\n  saved {OUT}")

if __name__ == "__main__":
    main()
