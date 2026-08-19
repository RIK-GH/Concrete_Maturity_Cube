"""
t8_core_logs.py  (TASK T5/T6 re-validation with REAL core-temperature logs)

Consumes the raw sensor logs (2-min sampling) supplied as:
  CUBE_TC-Mockup.xlsx   : datetime + TC-A/B/C-1..4 (thermocouples) + air/room refs
  SLAB_FBG-Mockup.xlsx  : Days + FBG 1..7 (core) + Ambient Temp + RH   (this is Exp#2)

Steps:
  1. Build the CORE temperature history for each member (cube A/B/C mean of TC-x-1..3
     - sensor 4 is an anomalous edge sensor; slab mean of FBG 1..7) and the matching
     AMBIENT history (cube TC-Air-In; slab 'Ambient Temp').
  2. Integrate maturity along the real history with the trapezoidal integrators in
     maturity.py:  Nurse-Saul M, Arrhenius equivalent age t_eq, degree-day DD.
  3. VALIDATE against the paper: slab Exp#2 core t_eq at 1/3/7/14 d must reproduce
     Table 3 (28.86 / 80.19 / 163.63 / 310.15 h).
  4. Emit exact CORE and AMBIENT maturity at each strength age -> core_maturity.csv,
     used by t5t6_slab.py (core-exact mode) to replace the isothermal reconstruction.
"""
import os
import numpy as np
import pandas as pd
from config import ROOT_DIR, OUT_DIR, CSV_ENC, EA_DEFAULT
import maturity as mat

CUBE_XLSX = os.path.join(ROOT_DIR, "CUBE_TC-Mockup.xlsx")
# real field logs (2-min sampling, ~28 days each):
EXP1_XLSX = os.path.join(ROOT_DIR, "SLAB_Exp_1.xlsx")             # Exp#1 (summer, FBG #1..10)
SLAB_DIR = os.path.join(ROOT_DIR, "SLAB_Exp_2")                   # Exp#2 (autumn)
FBGA_CSV = os.path.join(SLAB_DIR, "A", "FBGA_IoT.csv")           # Exp#2 core (FBG 1..7) + IoT
FBGC_CSV = os.path.join(SLAB_DIR, "C", "FBGC_IoT.csv")           # Exp#2 AD/SC control specimens
# paper Table 3 equivalent age (core, hours) for validation
PAPER_TE = {"Exp1": {3: 123.83, 7: 266.22, 28: 947.59},
            "Exp2": {1: 28.86, 3: 80.19, 7: 163.63, 14: 310.15, 28: 572.11}}
AGES_D = [1, 3, 7, 14, 28]
pd.set_option("display.width", 200)


def _read_csv(path):
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")


def mean_early_temp(time_h, temp_c, window_h=72.0):
    """Mean temperature over the first `window_h` hours - the crossover-relevant early
    thermal exposure (analogue of the isothermal curing temperature for lab literature)."""
    time_h = np.asarray(time_h, float); temp_c = np.asarray(temp_c, float)
    m = np.isfinite(time_h) & np.isfinite(temp_c) & (time_h <= window_h)
    return float(np.nanmean(temp_c[m])) if m.any() else float(np.nanmean(temp_c))


def maturity_at_ages(time_h, temp_c, ages_d, ea=EA_DEFAULT):
    """Return dict age_d -> (M_NS, t_eq, DD), interpolated onto the cumulative curves."""
    time_h = np.asarray(time_h, float)
    temp_c = np.asarray(temp_c, float)
    keep = np.isfinite(time_h) & np.isfinite(temp_c)   # drop trailing/blank samples
    time_h, temp_c = time_h[keep], temp_c[keep]
    m_ns = mat.integrate_nurse_saul(time_h, temp_c)
    t_eq = mat.integrate_eq_age(time_h, temp_c, ea=ea)
    dd = mat.integrate_degree_day(time_h, temp_c)
    out = {}
    for a in ages_d:
        th = a * 24.0
        if th <= time_h[-1]:
            out[a] = (np.interp(th, time_h, m_ns), np.interp(th, time_h, t_eq),
                      np.interp(th, time_h, dd), False)
        else:
            # extend beyond the log at the last observed (near-ambient) temperature
            dt_ext = th - time_h[-1]
            t_last = float(np.mean(temp_c[-30:]))
            out[a] = (m_ns[-1] + mat.nurse_saul_iso(t_last, dt_ext),
                      t_eq[-1] + mat.eq_age_iso(t_last, dt_ext, ea=ea),
                      dd[-1] + mat.degree_day_iso(t_last, dt_ext), True)
    return out


def load_cube():
    """Members 1-3 of each mix stood in the indoor laboratory; member 4 stood outdoors
    (its channel tracks TC-Air-Out, corr ~0.45-0.49 vs ~0.2 for members 1-3, and mix B's
    outdoor member peaks at 46.3 C under May sun). Indoor core history = mean of channels
    1-3; outdoor core history = channel 4."""
    df = pd.read_excel(CUBE_XLSX, sheet_name=0)
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    ok = dt.notna()
    df, dt = df[ok].reset_index(drop=True), dt[ok].reset_index(drop=True)
    time_h = (dt - dt.iloc[0]).dt.total_seconds().values / 3600.0
    cores, cores_out = {}, {}
    for cube, wc in [("A", 0.555), ("B", 0.50), ("C", 0.60)]:
        cols = [f"TC-{cube}-{i}" for i in (1, 2, 3)]
        cores[wc] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).values
        cores_out[wc] = pd.to_numeric(df[f"TC-{cube}-4"], errors="coerce").values
    amb = pd.to_numeric(df["TC-Air-In"], errors="coerce").values
    amb_out = pd.to_numeric(df["TC-Air-Out"], errors="coerce").values
    return time_h, cores, cores_out, amb, amb_out


def load_exp1():
    """Exp#1 (summer) real field log: FBG #1..10 core + Ambient Temp."""
    df = pd.read_excel(EXP1_XLSX, sheet_name=0)
    days = pd.to_numeric(df["days"], errors="coerce")
    fbg = [c for c in df.columns if "FBG" in str(c)]
    core = df[fbg].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    amb = pd.to_numeric(df["Ambient Temp"], errors="coerce")
    m = days.notna() & core.notna()
    return days[m].values * 24.0, core[m].values, amb[m].values, {}, "real SLAB_Exp_1 (summer, 28 d)"


def load_exp2():
    """Exp#2 (autumn) real field logs: FBGA core (FBG 1..7) + IoT ambient + AD/SC specimens."""
    a = _read_csv(FBGA_CSV)
    days = pd.to_numeric(a["days"], errors="coerce")
    fbg = [c for c in a.columns if str(c).strip().upper().startswith("FBG")]
    core = a[fbg].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    tcol = [c for c in a.columns if "Temperature" in str(c)]
    amb = pd.to_numeric(a[tcol[-1]], errors="coerce")
    m = days.notna() & core.notna()
    time_h = days[m].values * 24.0
    core, amb = core[m].values, amb[m].values
    spec = {}
    if os.path.exists(FBGC_CSV):
        c = _read_csv(FBGC_CSV)
        n = min(len(c), len(time_h))
        for key, col in [("AD", "Air Specimen"), ("SC", "Standard Specimen")]:
            if col in c.columns:
                spec[key] = (time_h[:n], pd.to_numeric(c[col], errors="coerce").values[:n])
    return time_h, core, amb, spec, "real SLAB_Exp_2 (autumn, 28 d)"


def main():
    print("=" * 78)
    print("TASK T8 - maturity recomputed from REAL core-temperature logs")
    print("=" * 78)

    # ---------- SLAB Exp#1 & Exp#2 validation vs paper Table 3 ----------
    slab_rows = []
    for exp, loader, wc in [("Exp1", load_exp1, 0.64), ("Exp2", load_exp2, 0.44)]:
        st, score, samb, spec, src = loader()
        print(f"\n[SLAB {exp}]  source = {src}; {len(st)} samples, {st[-1]/24:.1f} d, "
              f"core peak {np.nanmax(score):.1f} C @ {st[np.nanargmax(score)]/24:.2f} d, "
              f"ambient mean {np.nanmean(samb):.1f} C")
        core_m = maturity_at_ages(st, score, AGES_D)
        amb_m = maturity_at_ages(st, samb, AGES_D)
        spec_m = {k: maturity_at_ages(t, v, AGES_D) for k, (t, v) in spec.items()}
        Tearly_core = mean_early_temp(st, score)
        Tearly_amb = mean_early_temp(st, samb)
        if spec:
            print("  control specimens (real temps): "
                  + ", ".join(f"{k} mean {np.nanmean(v):.1f}C" for k, (t, v) in spec.items()))
        print(f"  early(72h) temp: core {Tearly_core:.1f}C, ambient {Tearly_amb:.1f}C")
        print("  age  core_teq  paper_teq   err%   | amb_teq  core/amb")
        for a in AGES_D:
            ct, at = core_m[a][1], amb_m[a][1]
            pt = PAPER_TE[exp].get(a)
            es = f"{100*(ct-pt)/pt:+6.1f}%" if pt else "    -  "
            ps = f"{pt:8.2f}" if pt else "    -   "
            print(f"  {a:>3}d  {ct:7.1f}  {ps}  {es}  | {at:7.1f}   {ct/at:5.2f}")
            r = dict(member=f"slab_{exp}", wc=wc, age_d=a,
                     M_NS_core=core_m[a][0], teq_core=ct, DD_core=core_m[a][2],
                     M_NS_amb=amb_m[a][0], teq_amb=at, DD_amb=amb_m[a][2], ratio_teq=ct/at,
                     Tearly_core=Tearly_core, Tearly_amb=Tearly_amb)
            for k in spec_m:
                r[f"teq_{k}"] = spec_m[k][a][1]
            slab_rows.append(r)

    # ---------- Ea SENSITIVITY (paper fixes Ea=38,300; prompt asks for sensitivity) ----------
    print("\n[Ea sensitivity]  core equivalent age vs activation energy (real logs)")
    print("  Ea(J/mol):     30000    35000    38300    45000    55000   (%chg @28d vs 38300)")
    for exp, loader in [("Exp1", load_exp1), ("Exp2", load_exp2)]:
        st, score, _, _, _ = loader()
        keep = np.isfinite(st) & np.isfinite(score)
        st2, sc2 = st[keep], score[keep]
        te28 = {}
        for ea in (30000, 35000, 38300, 45000, 55000):
            te = mat.integrate_eq_age(st2, sc2, ea=ea)
            te28[ea] = np.interp(28 * 24, st2, te)
        base = te28[38300]
        row = "  ".join(f"{te28[ea]:8.1f}" for ea in (30000, 35000, 38300, 45000, 55000))
        print(f"  {exp}: {row}   ({100*(te28[55000]-base)/base:+.0f}%..{100*(te28[30000]-base)/base:+.0f}%)")
    print("  -> Ea sensitivity is CONDITION-DEPENDENT: large for the hot-cured Exp#1")
    print("     (T>Tref, up to ~20% for +-20% Ea) but small for the cool Exp#2 (T~Tref, ~5%),")
    print("     with OPPOSITE sign - the classic Arrhenius behaviour. Calibrating Ea therefore")
    print("     matters most for hot-weather/massive members; near 20C it is nearly irrelevant.")

    # ---------- CUBE core vs ambient maturity (indoor members 1-3; outdoor member 4) ----------
    ct2, cores, cores_out, camb, camb_out = load_cube()
    print(f"\n[CUBE]  {len(ct2)} samples, {ct2[-1]/24:.1f} d, ambient(air-in) mean "
          f"{np.nanmean(camb):.1f} C, ambient(air-out) mean {np.nanmean(camb_out):.1f} C")
    cube_rows = []
    Tearly_camb = mean_early_temp(ct2, camb)
    Tearly_camb_out = mean_early_temp(ct2, camb_out)
    for wc, core in cores.items():
        pk = np.nanmax(core); tpk = ct2[np.nanargmax(core)] / 24
        cm = maturity_at_ages(ct2, core, AGES_D)
        am = maturity_at_ages(ct2, camb, AGES_D)
        Tearly_core = mean_early_temp(ct2, core)
        print(f"  w/c {wc} indoor : core peak {pk:.1f} C @ {tpk:.2f} d, early(72h) core {Tearly_core:.1f}C")
        for a in AGES_D:
            cube_rows.append(dict(member="cube", wc=wc, age_d=a,
                                  M_NS_core=cm[a][0], teq_core=cm[a][1], DD_core=cm[a][2],
                                  M_NS_amb=am[a][0], teq_amb=am[a][1], DD_amb=am[a][2],
                                  ratio_teq=cm[a][1] / am[a][1],
                                  Tearly_core=Tearly_core, Tearly_amb=Tearly_camb))
    for wc, core in cores_out.items():
        pk = np.nanmax(core); tpk = ct2[np.nanargmax(core)] / 24
        cm = maturity_at_ages(ct2, core, AGES_D)
        am = maturity_at_ages(ct2, camb_out, AGES_D)
        Tearly_core = mean_early_temp(ct2, core)
        print(f"  w/c {wc} outdoor: core peak {pk:.1f} C @ {tpk:.2f} d, early(72h) core {Tearly_core:.1f}C")
        for a in AGES_D:
            cube_rows.append(dict(member="cube_out", wc=wc, age_d=a,
                                  M_NS_core=cm[a][0], teq_core=cm[a][1], DD_core=cm[a][2],
                                  M_NS_amb=am[a][0], teq_amb=am[a][1], DD_amb=am[a][2],
                                  ratio_teq=cm[a][1] / am[a][1],
                                  Tearly_core=Tearly_core, Tearly_amb=Tearly_camb_out))

    # ---------- write exact maturities for both slabs & cube ----------
    allrows = pd.DataFrame(slab_rows + cube_rows)
    allrows.to_csv(os.path.join(OUT_DIR, "core_maturity.csv"), index=False, encoding=CSV_ENC)

    print("\n[Ambient->core equivalent-age ratio at early age] (was estimated 2.34x from peak-iso)")
    early = allrows[allrows.age_d.isin([1, 3])].groupby(["member", "age_d"]).ratio_teq.mean()
    print(early.to_string())
    print("\nWritten -> core_maturity.csv (slab_Exp1, slab_Exp2, cube)")


if __name__ == "__main__":
    main()
