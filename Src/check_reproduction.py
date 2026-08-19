"""
check_reproduction.py  -  assert that this run reproduces the published numbers.

Run after run_all.py. Compares the pipeline's own outputs against every quantitative
claim the manuscript makes that is derivable from Data/, and prints a pass/fail table.
A reader can use it to confirm the deposit reproduces the paper; the authors use it as a
regression test.

    uv run python check_reproduction.py

Exit status is 0 only if every check passes.
"""
import os
import sys
import numpy as np
import pandas as pd

from config import OUT_DIR, SI_DIR, CSV_ENC, RUN_MODE

R = lambda f: pd.read_csv(os.path.join(OUT_DIR, f), encoding=CSV_ENC)
results = []


def check(label, got, want, tol, unit=""):
    """Record one comparison. Anything not reducible to a single number fails loudly
    rather than silently passing (e.g. a duplicated index returning a Series)."""
    if isinstance(got, pd.Series):
        got = got.iloc[0] if len(got) == 1 else None
    try:
        got = None if got is None else float(got)
    except (TypeError, ValueError):
        got = None
    ok = got is not None and abs(got - want) <= tol
    results.append((label, got, want, tol, unit, ok))


def main():
    print("=" * 88)
    print(f"Reproduction check   (run mode: {RUN_MODE})")
    print("=" * 88)

    # ---- Section 2: data inventory -------------------------------------------------
    m = pd.read_parquet(os.path.join(OUT_DIR, "master.parquet"))
    check("Table 1  total records", len(m), 732, 0)
    check("Table 1  unique records (732 - lit55 subset)",
          len(m) - (m.dataset == "lit55").sum(), 677, 0)
    check("Sec 2.6  literature mix families (lit503 + lit119)",
          m[m.dataset.isin(["lit503", "lit119"])].mix_family.nunique(), 73, 0)
    for ds, n in [("cube", 33), ("slab", 22), ("lit119", 119), ("lit55", 55), ("lit503", 503)]:
        check(f"Table 1  {ds} records", (m.dataset == ds).sum(), n, 0)

    # ---- Section 2.4: cube strengths ------------------------------------------------
    c = m[m.dataset == "cube"]
    g = lambda wc, cur, age: c[(c.WC == wc) & (c.curing_type == cur) &
                               (c.age_day == age)].f_c.iloc[0]
    check("Sec 3.13 core indoor  W/C 0.50", g(0.50, "core_indoor", 28), 29.9, 0.05, "MPa")
    check("Sec 3.13 core outdoor W/C 0.50", g(0.50, "core_outdoor", 28), 24.8, 0.05, "MPa")
    check("Sec 3.13 core mean    W/C 0.50",
          (g(0.50, "core_indoor", 28) + g(0.50, "core_outdoor", 28)) / 2, 27.35, 0.06, "MPa")
    check("Sec 2.4  water 28 d W/C 0.50", g(0.50, "water", 28), 30.2, 0.05, "MPa")
    check("Sec 2.4  water 14 d W/C 0.555 (reversal high)", g(0.555, "water", 14), 31.3, 0.05, "MPa")
    check("Sec 2.4  water 28 d W/C 0.555 (reversal low)", g(0.555, "water", 28), 27.2, 0.05, "MPa")
    check("r11 fix  air   3 d W/C 0.50", g(0.50, "air", 3), 18.9, 0.05, "MPa")
    check("r11 fix  water 3 d W/C 0.50", g(0.50, "water", 3), 19.3, 0.05, "MPa")

    # ---- Section 3.7: sensor-free refit quoted in the text --------------------------
    sf = R("hyperbolic_fits_sensorfree.csv").set_index("mix_id")
    check("Sec 3.7  refit Su  (W/C 0.50)", sf.loc["cube_wc500", "Su"], 27.4, 0.06, "MPa")
    check("Sec 3.7  refit k   (W/C 0.50)", sf.loc["cube_wc500", "k"], 0.0212, 0.0002, "/degC h")
    check("Sec 2.4  refit Su  (W/C 0.60)", sf.loc["cube_wc600", "Su"], 23.5, 0.06, "MPa")

    # ---- Section 2.6: activation energy --------------------------------------------
    ea = R("ea_calibration.csv").Ea_Jmol.dropna()
    check("Sec 2.6  Ea min", ea.min(), 58600, 120, "J/mol")
    check("Sec 2.6  Ea max", ea.max(), 82900, 120, "J/mol")
    check("Sec 2.6  Ea mean", ea.mean(), 66400, 120, "J/mol")

    # ---- Section 3.8/3.9: learning results -----------------------------------------
    rt = R("results_table.csv")
    p503 = rt[(rt.experiment == "lit503") & (rt.cv_protocol.str.contains("family"))]
    check("Sec 3.9  pure GB RMSE (503 pool)",
          p503[p503.model == "GradientBoost"].RMSE.iloc[0], 16.1, 0.15, "MPa")
    check("Sec 3.9  hybrid RMSE (untuned)",
          p503[p503.model == "H1_residual"].RMSE.iloc[0], 11.9, 0.15, "MPa")
    tuned = p503[p503.model.str.contains("tuned")]
    check("Sec 3.9  hybrid RMSE (tuned)", tuned.RMSE.iloc[0], 11.3, 0.15, "MPa")
    check("Sec 3.10 hybrid R2   (tuned)", tuned.R2.iloc[0], 0.70, 0.02)
    l55 = rt[(rt.experiment == "lit55") & (rt.kind == "ours")]
    check("Sec 3.8  leakage demo RMSE (in-sample GB)",
          l55[l55.cv_protocol == "in-sample"].RMSE.iloc[0], 0.105, 0.02, "MPa")

    # ---- Section 3.10: screening surrogate on the whole pool ------------------------
    x = m[m.WC.notna() & m.t_eq.notna() & m.f_c.notna()]
    pred = (23.7 / x.WC - 11.0) * 0.018 * x.t_eq / (1 + 0.018 * x.t_eq)
    r2 = 1 - ((x.f_c - pred) ** 2).sum() / ((x.f_c - x.f_c.mean()) ** 2).sum()
    check("Sec 3.10 surrogate R2 across the pool", r2, 0.44, 0.02)
    check("Sec 3.10 surrogate RMSE across the pool",
          np.sqrt(((x.f_c - pred) ** 2).mean()), 14.0, 0.4, "MPa")

    # ---- Section 3.11: decision layer ----------------------------------------------
    d = R("decision_exceedance.csv").set_index("te_h")
    check("Sec 3.11 S_pred at 168 h", d.loc[168, "S_pred"], 26.8, 0.06, "MPa")
    check("Sec 3.11 P(S>=2/3 fck) at 168 h", d.loc[168, "P>=KCS_2/3fck"], 0.97, 0.01)
    check("Sec 3.11 S_pred at 48 h (>14 MPa floor)", d.loc[48, "S_pred"], 16.8, 0.06, "MPa")
    check("Sec 3.11 P(S>=14 MPa) at 72 h", d.loc[72, "P>=KCS_14MPa"], 0.98, 0.01)

    # ---- Section 3.12 / Table 3: autogenous valley ---------------------------------
    # The FBG strain record is released on request, not deposited, so these four checks
    # are unavailable to a reader of the public package. Skipped, not failed.
    skipped = []
    if os.path.exists(os.path.join(OUT_DIR, "strain_valley.csv")):
        v = R("strain_valley.csv").set_index("mix")
        check("Table 3  valley A (W/C 0.555)", v.loc["A", "valley_raw_ue"], -149.7, 0.2, "ue")
        check("Table 3  valley B (W/C 0.50)", v.loc["B", "valley_raw_ue"], -292.8, 0.2, "ue")
        check("Table 3  valley C (W/C 0.60)", v.loc["C", "valley_raw_ue"], -40.8, 0.2, "ue")
        A = np.vstack([v.eff_WC.values, np.ones(len(v))]).T
        coef, *_ = np.linalg.lstsq(A, v.valley_raw_ue.values, rcond=None)
        check("Sec 3.12 valley-vs-W/C slope", coef[0], 2523, 30, "ue per W/C")
    else:
        skipped.append("Table 3 autogenous valley (4 checks) - FBG strain record is "
                       "request-only and not part of the public deposit")

    # ---- Table 3 / Sec 2.1: cube peak temperatures from the deposited log ----------
    s7 = os.path.join(SI_DIR, "SI_Table_S7_core_temperature.csv")
    if os.path.exists(s7):
        t = pd.read_csv(s7, encoding=CSV_ENC)
        for mix, want in [("A", 41.02), ("B", 42.48), ("C", 38.34)]:
            peaks = [t[f"TC-{mix}-{i}_C"].max() for i in (1, 2, 3, 4)]
            check(f"Table 3  T_peak mix {mix} (mean of 4 members)",
                  float(np.mean(peaks)), want, 0.01, "degC")

    # ---- Table 7 / SI S5: member-size sweep ---------------------------------------
    # produced by vs_sweep_postproc.py when ABAQUS output is available; otherwise read the
    # deposited table, so the check covers Table 7 in both run modes
    sw_path = os.path.join(OUT_DIR, "vs_sweep_050.csv")
    s5 = os.path.join(SI_DIR, "SI_Table_S5_member_size_sweep.csv")
    if os.path.exists(sw_path) or os.path.exists(s5):
        if os.path.exists(sw_path):
            sw = R("vs_sweep_050.csv")
        else:
            sw = pd.read_csv(s5, encoding=CSV_ENC)
        sw = sw[sw.mix_wc.round(3) == 0.50].set_index("L_mm")
        check("Table 7  late-age ratio at L=400 mm",
              sw.loc[400, "strength_ratio_late14d"], 0.915, 0.006)
        check("Table 7  maturity ratio at L=400 mm", sw.loc[400, "maturity_ratio"], 1.296, 0.003)
        check("Table 7  core Su at L=400 mm", sw.loc[400, "Su_core_MPa"], 24.4, 0.06, "MPa")

    # ---- report --------------------------------------------------------------------
    w = max(len(r[0]) for r in results)
    print(f"\n  {'check':<{w}}  {'got':>10}  {'expected':>10}  {'tol':>8}  unit")
    print("  " + "-" * (w + 44))
    for label, got, want, tol, unit, ok in results:
        gs = "n/a" if got is None else f"{float(got):.4g}"
        print(f"  {'PASS' if ok else 'FAIL'} {label:<{w - 5}} {gs:>10}  {want:>10.4g}  "
              f"{tol:>8.4g}  {unit}")
    bad = [r for r in results if not r[5]]
    print(f"\n  {len(results) - len(bad)}/{len(results)} checks passed.")
    for sk in skipped:
        print(f"  SKIP  {sk}")
    if bad:
        print("  Failing checks:")
        for r in bad:
            print(f"    - {r[0]}: got {r[1]}, expected {r[2]} +-{r[3]}")
        return 1
    print("  This run reproduces every published number that Data/ can supply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
