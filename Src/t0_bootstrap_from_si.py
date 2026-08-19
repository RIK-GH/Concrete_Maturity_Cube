"""
t0_bootstrap_from_si.py  -  rebuild Data/ from the public deposit alone.

Why this exists
---------------
The authors' pipeline starts by parsing raw source files (a third-party literature
workbook, the field-slab logs of the prior publication, and the cube's thermocouple and
FBG spreadsheets). Two of those cannot be redistributed, so a reader who downloads the
public package does not have them.

Everything the manuscript reports is nevertheless contained in the deposited tables, and
this script converts them back into the intermediate products the rest of the pipeline
consumes. After it runs, stages t2 onward behave exactly as they do in the authors' tree
and produce the same numbers.

    SI_Table_S1  ->  Data/master.parquet  (+ per-subset CSVs)   the 732-record pool
    SI_Table_S3  ->  Data/ea_calibration.csv                    per-binder Ea
    SI_Table_S4  ->  Data/core_maturity.csv                     log-integrated maturity
    SI_Table_S5  ->  Data/vs_sweep_050.csv                      FE member-size sweep
    SI_Table_S6  ->  Data/strain_core_series.csv                cube autogenous strain
                     (request-only: absent from the public deposit, so this step is
                      skipped and the valley stage skips with it)

The derived model features (sqrt_DD, log_time, t_eq_day, hydration degrees) are recomputed
here with the same functions t1_build_master uses, not copied, so they stay consistent with
maturity.py. The two columns of master.parquet that are absent from S1 (WB, SP) are created
empty on purpose: they were dropped from the deposit as unreliable (inconsistent units
across sources) and are not model features -- see ml_utils.FEATURES_PAPER.
"""
import os
import numpy as np
import pandas as pd

import maturity as mat
from config import OUT_DIR, SI_DIR, RESTRICTED_DIR, CSV_ENC

S1 = os.path.join(SI_DIR, "SI_Table_S1_strength_maturity.csv")
S3 = os.path.join(SI_DIR, "SI_Table_S3_activation_energy.csv")
S4 = os.path.join(SI_DIR, "SI_Table_S4_core_vs_ambient_maturity.csv")
S5 = os.path.join(SI_DIR, "SI_Table_S5_member_size_sweep.csv")
S6_NAME = "SI_Table_S6_autogenous_strain.csv"
S6 = next((q for q in (os.path.join(RESTRICTED_DIR, S6_NAME),
                       os.path.join(SI_DIR, S6_NAME)) if os.path.exists(q)),
          os.path.join(SI_DIR, S6_NAME))

MASTER_COLS = ["dataset", "mix_id", "mix_family", "ref", "temp_source", "curing_type",
               "curing_C", "age_day", "time_h", "M_NS", "t_eq", "DD", "WC", "WB", "C",
               "binder", "RH", "Sa", "SP", "fck", "peakT", "peak_day",
               "provisional_maturity", "f_c"]

# S4 uses publication-facing column names; core_maturity.csv uses the pipeline's.
S4_TO_CM = {"WC": "wc", "age_day": "age_d", "t_eq_core": "teq_core",
            "M_NS_ambient": "M_NS_amb", "t_eq_ambient": "teq_amb", "DD_ambient": "DD_amb",
            "ratio_teq_core_amb": "ratio_teq", "Tearly_core_C": "Tearly_core",
            "Tearly_amb_C": "Tearly_amb", "t_eq_AD": "teq_AD", "t_eq_SC": "teq_SC"}


def need(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"\n  Missing deposit file: {os.path.basename(path)}\n"
            f"  Expected in: {SI_DIR}\n"
            f"  Download the complete Supplementary_Dataset/ folder next to Src/.\n")
    return path


def build_master():
    s1 = pd.read_csv(need(S1), encoding=CSV_ENC)
    m = s1.rename(columns={"cement": "C", "maturity_provisional": "provisional_maturity"})
    for c in ("WB", "SP"):
        if c not in m.columns:
            m[c] = np.nan
    missing = [c for c in MASTER_COLS if c not in m.columns]
    if missing:
        raise SystemExit(f"  SI_Table_S1 is missing expected columns: {missing}")
    m = m[MASTER_COLS].copy()

    # same derivation as t1_build_master.py
    m["sqrt_DD"] = np.sqrt(m["DD"].clip(lower=0))
    m["log_time"] = np.log(m["time_h"].clip(lower=1e-3))
    m["t_eq_day"] = m["t_eq"] / 24.0
    m["alpha_bazant"] = mat.hydration_degree_bazant(m["t_eq_day"])
    m["alpha_fh"] = mat.hydration_degree_fh(m["t_eq_day"])

    m.to_parquet(os.path.join(OUT_DIR, "master.parquet"), index=False)
    for ds, g in m.groupby("dataset"):
        g.to_csv(os.path.join(OUT_DIR, f"{ds}.csv"), index=False, encoding=CSV_ENC)
    return len(m), m


def build_core_maturity():
    s4 = pd.read_csv(need(S4), encoding=CSV_ENC).rename(columns=S4_TO_CM)
    s4.to_csv(os.path.join(OUT_DIR, "core_maturity.csv"), index=False, encoding=CSV_ENC)
    return len(s4)


def passthrough(src, dst, renames=None, where=None):
    if not os.path.exists(src):
        return None
    d = pd.read_csv(src, encoding=CSV_ENC)
    if renames:
        d = d.rename(columns=renames)
    if where is not None:
        d = d[where(d)].reset_index(drop=True)
    d.to_csv(os.path.join(OUT_DIR, dst), index=False, encoding=CSV_ENC)
    return len(d)


def main():
    print("=" * 78)
    print("STAGE T0 - rebuilding Data/ from Supplementary_Dataset/ (deposit mode)")
    print("=" * 78)

    n, m = build_master()
    print(f"  master.parquet                 {n:6d} rows   "
          f"({m.dataset.nunique()} subsets, {m.mix_family.nunique()} mix families)")
    for ds, g in m.groupby("dataset"):
        print(f"    {ds}.csv{'':>{max(0, 22-len(ds))}} {len(g):6d} rows")

    print(f"  core_maturity.csv              {build_core_maturity():6d} rows")

    for src, dst, ren, where in [
        (S3, "ea_calibration.csv", None, None),
        # S5 holds both sweeps; vs_sweep_postproc.py writes one file per mix, so split.
        (S5, "vs_sweep_050.csv", None, lambda d: d.mix_wc.round(3) == 0.50),
        (S5, "vs_sweep_060.csv", None, lambda d: d.mix_wc.round(3) == 0.60),
        (S6, "strain_core_series.csv",
         {"elapsed_h_from_casting_zero": "elapsed_h", "core_strain_ue": "strain_ue"}, None),
    ]:
        n = passthrough(src, dst, ren, where)
        label = os.path.basename(dst)
        if n is None:
            note = ("   [request-only, not deposited - valley stage will skip]"
                    if dst == "strain_core_series.csv" else "   [absent - skipped]")
            print(f"  {label:30s} {'-':>6}{note}")
        else:
            print(f"  {label:30s} {n:6d} rows")

    print("\n  Data/ is ready. Continue with the compute stages (see run_all.py).")


if __name__ == "__main__":
    main()
