"""
t2b_sensorfree_refit.py  -  Cube hyperbolic refit EXCLUDING the sensor-bearing
28-day companion cylinders.

Why this exists
---------------
Section 2 of the manuscript discloses that the 28-day companion cylinders carried an
embedded FBG sensor and that insufficient compaction around it depressed their measured
strength (visible where a 28-day companion falls below its own 14-day value, e.g. the
W/C 0.555 water-cured control, 31.3 -> 27.2 MPa). The manuscript then reports a refit of
each per-mix hyperbolic on the SENSOR-FREE data only, and quotes numbers from it
(Section 2.4: 'W/C 0.60 Su 22.2 -> 23.5 MPa'; Section 3.7: 'Su = 27.3 MPa, k = 0.0229').

Those refit values were previously computed ad hoc and never persisted, so they could not
be reproduced from the shipped data package. This script recomputes them with the same
fitting routine used for the main fits (t2_hyperbolic.fit_mix) and writes
    Data/hyperbolic_fits_sensorfree.csv
alongside the full-data Data/hyperbolic_fits.csv, leaving the latter untouched.

Definition of 'sensor-free' (as stated in the manuscript)
    drop  : the 28-day air-dried and water-cured COMPANION cylinders
    retain: the 1-14-day companions (no sensor) and both 28-day DRILLED CORES
            (cores carry no sensor and are the reliable 28-day in-place strength)
"""
import os
import numpy as np
import pandas as pd
from config import OUT_DIR, CSV_ENC
from t2_hyperbolic import fit_mix

MASTER = os.path.join(OUT_DIR, "master.parquet")
OUT = os.path.join(OUT_DIR, "hyperbolic_fits_sensorfree.csv")
COMPANION = ("air", "water")          # cast companion cylinders (sensor-bearing at 28 d)


def main():
    m = pd.read_parquet(MASTER)
    cube = m[m.dataset == "cube"].copy()
    drop = (cube.age_day == 28) & (cube.curing_type.isin(COMPANION))
    kept = cube[~drop]

    print("=" * 78)
    print("Cube hyperbolic refit on sensor-free strength data")
    print("=" * 78)
    print(f"  cube rows {len(cube)}  ->  dropped {int(drop.sum())} sensor-bearing 28-d "
          f"companions  ->  kept {len(kept)}")

    rows = []
    for mid, g in kept.groupby("mix_id"):
        full = cube[cube.mix_id == mid]
        f_new = fit_mix(g.t_eq.values, g.f_c.values)
        f_old = fit_mix(full.t_eq.values, full.f_c.values)
        if f_new is None:
            print(f"  {mid}: refit failed (n={len(g)})")
            continue
        su, k, r2, rmse, n = f_new["Su"], f_new["k"], f_new["r2"], f_new["rmse"], f_new["n"]
        su0, k0 = (f_old["Su"], f_old["k"]) if f_old else (np.nan, np.nan)
        rows.append(dict(dataset="cube", mix_id=mid, WC=float(g.WC.iloc[0]),
                         Su=su, k=k, r2=r2, rmse=rmse, n=n,
                         Su_fulldata=su0, k_fulldata=k0))
        print(f"  {mid}: Su {su0:6.2f} -> {su:6.2f} MPa   k {k0:.4f} -> {k:.4f}   "
              f"(n {len(full)} -> {n}, R2 {r2:.3f})")

    d = pd.DataFrame(rows).sort_values("WC")
    d.to_csv(OUT, index=False, encoding=CSV_ENC)
    print(f"\nWritten -> {os.path.basename(OUT)}")

    # mean of the per-mix refit, quoted in Section 3.7 for the strength conversion
    if len(d):
        print(f"  refit mean over the three mixes: Su = {d.Su.mean():.1f} MPa, "
              f"k = {d.k.mean():.4f} per degC h")


if __name__ == "__main__":
    main()
