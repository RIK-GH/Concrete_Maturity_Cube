"""
t13_strain_valley.py  -  Autogenous-shrinkage valley depths from the cube FBG strain record.

The manuscript's Table 3 reports, per mix, the depth of the early autogenous-shrinkage
*valley*: the embedded (core) FBG strain falls from its casting zero to a minimum at
~18-22 h and then recovers. This script computes that depth from the raw record so the
tabulated numbers are reproducible, and writes
    Data/strain_valley.csv          per-mix valley depth and time
    Data/strain_core_series.csv     the 2-min core series (SI-ready, long format)

Conventions, all taken from the record itself and the supplier's label note
(see config.CUBE_STRAIN_EMB):
  * strain zero  : each mix's own first valid sample, which the supplier set 1-2 h before
                   completion of placement of that mix. The three mixes were placed in
                   sequence, so the series start at different clock times -- this is why
                   elapsed time is measured per mix, never from a common log origin.
  * channel      : EMBEDDED (core) only. The surface gauges were installed one day after
                   casting and so cannot see the early valley.
  * estimator    : minimum over the first 3 days. Reported both raw and after a 15-sample
                   (30-min) centred median, because the raw minimum can sit on a single
                   noise spike -- mix C's raw minimum is 2.8 ue deeper than its plateau.
  * sign         : negative = contraction. The gauges are temperature-compensated, so the
                   thermal component is already removed and the ordering is the autogenous
                   one (lower W/C -> deeper valley), not the thermal one.
"""
import os
import numpy as np
import pandas as pd
from config import OUT_DIR, SI_DIR, RESTRICTED_DIR, CSV_ENC, CUBE_STRAIN_EMB, HAVE_RAW

# The FBG strain record is request-only, so it is NOT in the public deposit. Look for it in
# Restricted/ (authors' tree, or a requester who received it), then in Supplementary_Dataset/
# in case a future release makes it public.
S6_NAME = "SI_Table_S6_autogenous_strain.csv"
S6_PATHS = [os.path.join(RESTRICTED_DIR, S6_NAME), os.path.join(SI_DIR, S6_NAME)]

MIX_WC = {"A": 0.555, "B": 0.50, "C": 0.60}     # effective W/C
WINDOW_D = 3.0                                   # 'initial 3 days' of Table 3
MEDIAN_N = 15                                    # 15 samples x 2 min = 30 min


def load_core():
    """Return ({mix: DataFrame(elapsed_h, strain_ue)}, {mix: strain-zero timestamp or None}).

    Prefers the source spreadsheet, which carries wall-clock timestamps; falls back to the
    deposited SI Table S6, whose elapsed_h is already referenced to each mix's own zero.
    """
    # In deposit mode the raw spreadsheet is ignored even when present, so that
    # CONCRETE_RUN_MODE=deposit really does reproduce a reader's environment.
    if HAVE_RAW and os.path.exists(CUBE_STRAIN_EMB):
        df = pd.read_excel(CUBE_STRAIN_EMB, sheet_name=0)
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        out, zero = {}, {}
        for m in MIX_WC:
            s = pd.to_numeric(df[f"{m}-I_strain_ue"], errors="coerce")
            ok = s.notna()
            if not ok.any():
                continue
            t0 = dt[ok].iloc[0]                   # this mix's strain zero
            zero[m] = t0
            h = (dt - t0).dt.total_seconds() / 3600.0
            out[m] = pd.DataFrame({"elapsed_h": h[ok].values, "strain_ue": s[ok].values})
        return out, zero

    table = next((q for q in S6_PATHS if os.path.exists(q)), None)
    if table is None:
        return None, None            # request-only and not present - caller skips
    print(f"  reading {os.path.basename(table)}")
    d = pd.read_csv(table, encoding=CSV_ENC)
    out, zero = {}, {}
    for m, g in d.groupby("mix"):
        out[m] = pd.DataFrame({"elapsed_h": g.elapsed_h_from_casting_zero.values,
                               "strain_ue": g.core_strain_ue.values})
        zero[m] = None                            # wall-clock times are not deposited
    return out, zero


def main():
    print("=" * 78)
    print("TASK T13 - autogenous-shrinkage valley from the cube FBG strain record")
    print("=" * 78)
    core, zero = load_core()
    if core is None:
        print("  The cube FBG strain record is available from the authors on reasonable\n"
              "  request and is not part of the public deposit, so the autogenous valley is\n"
              "  not recomputed here. Every other stage is unaffected; the resulting depths\n"
              "  are quoted in Table 3 of the manuscript.")
        return

    if all(v is not None for v in zero.values()):
        print("  strain zero per mix (supplier convention: 1-2 h before end of placement)")
        base = min(zero.values())
        for m in ("A", "B", "C"):
            print(f"    mix {m}: {zero[m]}  ({(zero[m]-base).total_seconds()/3600:+.2f} h)")
    else:
        print("  strain zero per mix: each series already starts at its own casting zero")

    rows, series = [], []
    print(f"\n  valley over the first {WINDOW_D:.0f} days (negative = contraction)")
    for m in ("A", "B", "C"):
        d = core[m]
        w = d[d.elapsed_h.between(0, WINDOW_D * 24)]
        i = w.strain_ue.idxmin()
        raw, t_raw = w.strain_ue.loc[i], w.elapsed_h.loc[i]
        med = w.strain_ue.rolling(MEDIAN_N, center=True, min_periods=1).median()
        j = med.idxmin()
        rows.append(dict(mix=m, eff_WC=MIX_WC[m],
                         valley_raw_ue=round(float(raw), 1),
                         valley_time_h=round(float(t_raw), 2),
                         valley_median30min_ue=round(float(med.loc[j]), 1),
                         n_samples=int(len(w))))
        print(f"    mix {m} (W/C {MIX_WC[m]}): raw {raw:8.1f} ue @ {t_raw:5.1f} h   "
              f"30-min median {med.loc[j]:8.1f} ue")
        s = core[m].copy(); s.insert(0, "mix", m)
        series.append(s)

    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(OUT_DIR, "strain_valley.csv"), index=False, encoding=CSV_ENC)
    pd.concat(series, ignore_index=True).round(3).to_csv(
        os.path.join(OUT_DIR, "strain_core_series.csv"), index=False, encoding=CSV_ENC)

    # linearity of valley depth in effective W/C (qualitative claim of Section 3.12)
    A = np.vstack([v.eff_WC.values, np.ones(len(v))]).T
    for lab, y in [("raw", v.valley_raw_ue.values), ("30-min median", v.valley_median30min_ue.values)]:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ c
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\n  valley depth vs effective W/C ({lab}): "
              f"slope {c[0]:.0f} ue per unit W/C, R2 = {r2:.4f}")

    print("\nWritten -> strain_valley.csv, strain_core_series.csv")


if __name__ == "__main__":
    main()
