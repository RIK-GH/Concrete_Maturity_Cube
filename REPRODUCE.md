# Reproducing the results

This deposit is self-contained. Downloading it and running one command reproduces every
number the manuscript reports. No raw source files, no solver licence and no path editing
are required.

## Quick start

```bash
# 1. get uv (a single static binary; https://docs.astral.sh/uv/)
#    macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows:      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv run python Src/run_all.py            # builds everything (~3-6 min on a laptop)
uv run python Src/check_reproduction.py # asserts the published numbers, prints a pass/fail table
```

`uv run` creates the environment from `pyproject.toml` and the pinned `uv.lock` (both at the
repository root) on first use, so the package versions are the ones the results were produced
with. Python 3.11-3.13. Run from the root as shown, or from `Src/` with the paths dropped --
`uv run` resolves the project either way.

Prefer plain pip? `python -m venv .venv && .venv/bin/pip install -r <(uv export --no-hashes)`,
or install the dependencies listed in `pyproject.toml`. Package versions then differ from the
lock file and the machine-learning figures may move in the last decimal.

Expected tail of the second command:

```
  39/39 checks passed.
  SKIP  Table 3 autogenous valley (4 checks) - FBG strain record is request-only and not
        part of the public deposit
  This run reproduces every published number that Data/ can supply.
```

The SKIP line is normal — see *Data released on request* below. With the restricted record
in place the count is 43/43.

If a check fails, the table names the quantity, what it got and what was expected — please
report it with that line.

## What runs, and in which mode

`config.py` detects the situation and `run_all.py` picks the stage list. You get
**deposit mode**; the authors' tree runs in **full mode**. Both produce identical numbers.

| | deposit mode (you) | full mode (authors) |
|---|---|---|
| Ingest | `t0_bootstrap_from_si.py` rebuilds `Data/` from `Supplementary_Dataset/` | `t8_core_logs.py` → `t1_build_master.py` parse the raw source files |
| Compute | identical: `t2` → `t2b` → `t3t4` → `t10` → `t5t6` → `t11` → `t9` → `t7` → `t12` → `t13` → `make_*` | identical |

Force the reader's path on the authors' machine with `CONCRETE_RUN_MODE=deposit` — that is
how the deposit is tested before release.

**Why two modes.** Two of the raw inputs cannot be redistributed: the literature workbook is
a third-party compilation, and the field-slab logs belong to the prior publication
(*Frontiers in Materials* **13**:1762995). Everything derived from them is deposited, so
`t0` reconstructs the intermediate products and the rest of the pipeline is untouched.
`SI_Table_S1` is written in the pipeline's own row order and at full numeric precision
specifically so that this reconstruction is exact — the grouped cross-validation assigns
folds by position, so a re-ordered or rounded table would shift the reported RMSE.

## Where each result comes from

| Manuscript item | Reproduced by | Output |
|---|---|---|
| Table 1 inventory (732 / 677 / 73 families) | `t0` or `t1` | `Data/master.parquet` |
| Table 3 peak temperatures | deposited log | `SI_Table_S7` |
| Table 3 autogenous valley | `t13_strain_valley.py` | `Data/strain_valley.csv` (needs the request-only strain record) |
| Table 4 / §2.6 activation energy | `t1` (full) / `SI_Table_S3` | `Data/ea_calibration.csv` |
| §3.7 sensor-free refit (`Su`, `k`) | `t2b_sensorfree_refit.py` | `Data/hyperbolic_fits_sensorfree.csv` |
| §3.8 leakage demonstration | `t3t4_hybrid_cv.py` | `Data/t3t4_cv_results.csv` |
| §3.9 / Table 11, 503-pool errors | `t3t4_hybrid_cv.py`, `t10_hpo.py` | `Data/t3t4_bigpool_results.csv`, `hpo_summary.csv` |
| §3.10 screening surrogate | `physics_prior.py` (checked directly) | `Data/surrogate_lookup.csv` |
| §3.11 conformal coverage + strike decision | `t7_uncertainty_decision.py` | `Data/decision_exceedance.csv` |
| Fig. 11 SHAP attribution | `t12_shap.py` | `Data/shap_importance.csv` |
| Table 5 / 8 in-place ratios | `t5t6_slab.py` | `Data/slab_validation.csv` |

## Data released on request

Two records are supplied by the data owners on request rather than deposited:

| Record | Why | Effect on this run |
|---|---|---|
| Cube embedded-FBG autogenous-strain series (30 700 samples) | Released on reasonable request by the data owner | `t13_strain_valley.py` prints a notice and skips; the four Table 3 valley checks are reported **SKIP**, not FAIL |
| Raw sensing logs of the two reference field slabs | Under the access terms of the prior publication (*Front. Mater.* **13**:1762995) | None — the slab strength and maturity values used here are deposited in `SI_Table_S1` and `SI_Table_S4` |

Everything derived from these records is reported in the article, and the code that derives
it is in this deposit, so the derivation is inspectable even without the raw series. To
request them, contact the corresponding author; drop the file into `Restricted/` and re-run,
and the check goes to 43/43.

## The one thing you cannot re-run

**Table 7 / `SI_Table_S5`, the finite-element member-size sweep.** The core-temperature
histories come from ABAQUS runs of the calibrated HETVAL/UMAT model, which needs the solver
and the mesh files. The resulting table is deposited, and `t0` loads it, so the numbers are
present and checked (`Table 7 …` rows in the check) — but `vs_sweep_postproc.py` and
`vs_sweep_compare_060.py` will report `[skip] file not found` unless you point
`CONCRETE_ABAQUS_DIR` at your own runs. Nothing else depends on the solver.

Two figure panels also read raw logs that are not deposited; `make_figures.py` skips them and
says so. This never affects a reported number, and `run_all.py` continues.

## Layout after a run

```
Supplementary_Dataset/   input: the deposited tables (S1-S5, S7, dictionary, provenance)
Restricted/              optional: request-only records, if you received them
Src/                     the pipeline
Data/                    output: everything the manuscript quotes, rebuilt
Doc/figures/             output: regenerated figures
```

`Data/` is created by the run; it is not part of the download.

## Provenance and licence

Literature records keep their source attributions (`SI_sources.csv`, `ref` column) and remain
the property of their respective authors. The cube and slab measurements are original data of
the authors. Code is MIT (`LICENSE-MIT.txt`); data and documentation are CC BY 4.0
(`LICENSE-CC-BY-4.0.txt`). Please cite the article and the data deposit — see `README.md`.
