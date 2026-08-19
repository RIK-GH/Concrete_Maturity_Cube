"""run_all.py - reproduce the whole pipeline end to end.

    uv run python run_all.py            # or: python run_all.py

Two modes, detected automatically (see config.py):

  full     the authors' tree, raw source files present in ROOT_DIR. Parses them from
           scratch: t8 integrates maturity from the raw 2-min logs, then t1 builds the
           732-record master pool.

  deposit  what you get by downloading the public package (Src/ + Supplementary_Dataset/).
           t0 rebuilds Data/ from the deposited tables instead, then every compute stage
           runs unchanged and produces the same numbers as the manuscript.

Force the reader's path with  CONCRETE_RUN_MODE=deposit  to verify the deposit is complete.
Stages that need ABAQUS output (the member-size sweep) are not part of either list; the
resulting table is deposited as SI Table S5. See REPRODUCE.md.
"""
import subprocess, sys, os
from config import RUN_MODE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# t8 runs FIRST in full mode so t1 can inject the log-integrated cube core maturity.
INGEST = {"full": ["t8_core_logs.py", "t1_build_master.py"],
          "deposit": ["t0_bootstrap_from_si.py"]}

COMPUTE = ["t2_hyperbolic.py", "t2b_sensorfree_refit.py", "t3t4_hybrid_cv.py",
           "t10_hpo.py", "t5t6_slab.py", "t11_tempweighted.py",
           "t9_calibration.py", "t7_uncertainty_decision.py", "t12_shap.py",
           "t13_strain_valley.py",
           "make_figures.py", "make_deliverables.py", "make_si_tables.py"]

STEPS = INGEST[RUN_MODE] + COMPUTE


def main():
    print(f"\n########## run mode: {RUN_MODE} ##########")
    if RUN_MODE == "deposit":
        print("Raw source files not found - rebuilding Data/ from Supplementary_Dataset/.")
    failed = []
    for s in STEPS:
        print(f"\n########## {s} ##########")
        r = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, s)], cwd=SCRIPT_DIR)
        if r.returncode != 0:
            failed.append(s)
            # make_figures needs raw logs for two panels; never fatal for the numbers
            if s == "make_figures.py":
                print(f"  [warn] {s} returned {r.returncode} - figures may be incomplete; "
                      f"reported numbers are unaffected.")
                continue
            print(f"\n  FAILED at {s} (exit {r.returncode}). Stopping.")
            return 1
    if failed:
        print(f"\nDone, with warnings from: {', '.join(failed)}")
    else:
        print("\nDone - all stages completed.")
    print("Outputs in ../Data/ ; regenerated deposit tables in ../Supplementary_Dataset/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
