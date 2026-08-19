"""
make_si_tables.py  -  regenerate the Supplementary_Dataset/ CSV tables from Data/.

Until r11 the SI tables S1-S4 were assembled by hand, so a correction in the pipeline
(e.g. the W/C 0.50 3-day strengths of r11) did not propagate to the submitted package.
This script derives them from the pipeline outputs instead, so `run_all.py` leaves the
SI package consistent with Data/ by construction.

Produced here:
    SI_Table_S1_strength_maturity.csv    from master.parquet          (732 rows)
    SI_Table_S2_hyperbolic_fits.csv      from hyperbolic_fits.csv
    SI_Table_S3_activation_energy.csv    from ea_calibration.csv
    SI_Table_S6_autogenous_strain.csv    from strain_core_series.csv  (t13)
        -> written to Restricted/, NOT to the deposit: the FBG strain record is
           released on request, not published (see README, Data availability).
    SI_Table_S7_core_temperature.csv     from CUBE_TC-Mockup.xlsx     (raw 2-min log)

Left alone (not derivable from Data/ alone):
    SI_Table_S4  core-vs-ambient maturity  (assembled from core_maturity.csv + slab logs)
    SI_Table_S5  member-size sweep         (written by vs_sweep_postproc.py / _compare_060)
    SI_sources, SI_data_dictionary, SI_README, surrogate_lookup
"""
import os
import pandas as pd
from config import OUT_DIR, CVS_DIR, CSV_ENC, CUBE_TC, RESTRICTED_DIR, HAVE_RAW

SI_DIR = os.path.join(CVS_DIR, "Supplementary_Dataset")
MIX_WC = {"A": 0.555, "B": 0.50, "C": 0.60}

S1_COLS = ["record_id", "dataset", "mix_id", "mix_family", "ref", "temp_source",
           "curing_type", "curing_C", "WC", "cement", "fly_ash", "SCM_other", "binder",
           "Sa", "RH", "age_day", "time_h", "M_NS", "t_eq", "DD", "f_c", "fck",
           "peakT", "peak_day", "maturity_provisional"]


def s1():
    """Write S1 from master.parquet, preserving the mix-composition columns.

    Two requirements shape this function.

    * **Exact reproducibility.** A reader rebuilds `master.parquet` from S1 (see
      t0_bootstrap_from_si.py), so S1 must carry the pipeline's *row order* and full
      numeric precision. The grouped cross-validation assigns folds by position, so a
      re-ordered table silently changes the reported RMSE; and rounding the maturity
      columns to 2 dp shifts it too. Both are therefore avoided here.
    * **No information loss.** `fly_ash` and `SCM_other` are published in S1 but are not
      retained by master.parquet. They are mix-level constants (verified: one value per
      `mix_id`), so they are joined back on `mix_id` rather than by row -- which also
      sidesteps the pooled literature's genuine replicate rows, where the same mix,
      condition and age appear more than once and no row key exists.
    """
    p = os.path.join(SI_DIR, "SI_Table_S1_strength_maturity.csv")
    m = pd.read_parquet(os.path.join(OUT_DIR, "master.parquet")).reset_index(drop=True)
    m = m.rename(columns={"C": "cement", "provisional_maturity": "maturity_provisional"})

    comp = None
    if os.path.exists(p):
        old = pd.read_csv(p, encoding=CSV_ENC)
        have = [c for c in ("fly_ash", "SCM_other") if c in old.columns]
        if have:
            g = old.groupby("mix_id")[have].nunique(dropna=False)
            if (g > 1).any().any():
                print("    [warn] fly_ash/SCM_other vary within a mix_id - not joined")
            else:
                comp = old.groupby("mix_id")[have].first().reset_index()

    if comp is not None:
        m = m.merge(comp, on="mix_id", how="left")
    for c in S1_COLS:
        if c not in m.columns:
            m[c] = pd.NA
    out = m[[c for c in S1_COLS if c != "record_id"]].copy()
    out.insert(0, "record_id", range(1, len(out) + 1))
    out.to_csv(p, index=False, encoding=CSV_ENC)   # full precision, pipeline row order
    kept = "with fly_ash/SCM_other" if comp is not None else "fly_ash/SCM_other unavailable"
    print(f"    written in pipeline row order at full precision ({kept})")
    return p, len(out)


def s2():
    h = pd.read_csv(os.path.join(OUT_DIR, "hyperbolic_fits.csv"), encoding=CSV_ENC)
    m = pd.read_parquet(os.path.join(OUT_DIR, "master.parquet"))
    ref = m.groupby(["dataset", "mix_id"])["ref"].first().reset_index()
    out = (h.merge(ref, on=["dataset", "mix_id"], how="left")
             .rename(columns={"Su": "Su_MPa", "k": "k_perMPaMaturity",
                              "r2": "R2", "rmse": "RMSE_MPa", "n": "n_points"}))
    out = out[["dataset", "mix_id", "ref", "Su_MPa", "k_perMPaMaturity",
               "R2", "RMSE_MPa", "n_points"]]
    out["Su_MPa"] = out.Su_MPa.round(2)
    out["R2"] = out.R2.round(4)
    out["RMSE_MPa"] = out.RMSE_MPa.round(3)
    p = os.path.join(SI_DIR, "SI_Table_S2_hyperbolic_fits.csv")
    out.to_csv(p, index=False, encoding=CSV_ENC)
    return p, len(out)


def s3():
    e = pd.read_csv(os.path.join(OUT_DIR, "ea_calibration.csv"), encoding=CSV_ENC)
    e["Ea_Jmol"] = pd.to_numeric(e.Ea_Jmol, errors="coerce").round(0)
    e["r2_arrhenius"] = pd.to_numeric(e.r2_arrhenius, errors="coerce").round(4)
    p = os.path.join(SI_DIR, "SI_Table_S3_activation_energy.csv")
    e.to_csv(p, index=False, encoding=CSV_ENC)
    return p, len(e)


def s7():
    """Raw core-thermocouple log, the temperature source for every result in the paper.

    Kept in WIDE form because, unlike the strain record, all twelve channels share one
    clock: the logger was started before the first mix was placed and ran continuously.
    Members 1-3 of each mix stood indoors, member 4 outdoors under field exposure.
    """
    if not (HAVE_RAW and os.path.exists(CUBE_TC)):
        return None, 0        # already deposited; nothing to regenerate without the source
    df = pd.read_excel(CUBE_TC, sheet_name=0)
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[dt.notna()].reset_index(drop=True)
    dt = dt[dt.notna()].reset_index(drop=True)
    out = pd.DataFrame({"datetime": dt.dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_h": ((dt - dt.iloc[0]).dt.total_seconds() / 3600.0).round(4)})
    for m in ("A", "B", "C"):
        for i in (1, 2, 3, 4):
            c = f"TC-{m}-{i}"
            out[f"{c}_C"] = pd.to_numeric(df[c], errors="coerce").round(3)
    for src, dst in [("TC-Air-In", "ambient_indoor_C"), ("TC-Air-Out", "ambient_outdoor_C"),
                     ("Humidity(%)-Indoor", "RH_indoor_pct"),
                     ("Humidity-Outdoor", "RH_outdoor_pct")]:
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce").round(3)
    p = os.path.join(SI_DIR, "SI_Table_S7_core_temperature.csv")
    out.to_csv(p, index=False, encoding=CSV_ENC)
    return p, len(out)


def s6():
    src = os.path.join(OUT_DIR, "strain_core_series.csv")
    if not os.path.exists(src):
        return None, 0
    d = pd.read_csv(src, encoding=CSV_ENC)
    d = d.rename(columns={"elapsed_h": "elapsed_h_from_casting_zero",
                          "strain_ue": "core_strain_ue"})
    d["eff_WC"] = d["mix"].map(MIX_WC)
    d = d[["mix", "eff_WC", "elapsed_h_from_casting_zero", "core_strain_ue"]]
    os.makedirs(RESTRICTED_DIR, exist_ok=True)
    p = os.path.join(RESTRICTED_DIR, "SI_Table_S6_autogenous_strain.csv")
    d.round(3).to_csv(p, index=False, encoding=CSV_ENC)
    return p, len(d)


def main():
    print("=" * 78)
    print("Regenerating Supplementary_Dataset/ tables from Data/")
    print("=" * 78)
    for fn in (s1, s2, s3, s6, s7):
        p, n = fn()
        if p is None:
            why = {"s6": "strain record is request-only",
                   "s7": "raw thermocouple log not present; the deposited table stands"}
            print(f"  [skip] {fn.__name__}: {why.get(fn.__name__, 'source not found')}")
        else:
            print(f"  {os.path.basename(p):42s} {n:6d} rows")
    print("\n  S4/S5 and the descriptive files are not regenerated here (see module docstring).")


if __name__ == "__main__":
    main()
