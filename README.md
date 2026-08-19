# Concrete_Maturity_Cube — Supplementary dataset and code package

Supporting materials for the manuscript **"Member-size dependence of in-place concrete
strength: sensor-monitored maturity for formwork removal"** (Park, Choe, Kim & Rhee;
under review, *Case Studies in Construction Materials*), which advances the framework
of Park, Choe, Kim & Rhee, *Frontiers in Materials* **13**:1762995 (2026).

**Package version: r11.0 (2026-07-26).** Changes from r10.2:

- **Cube FBG autogenous-strain record processed** (30 700 samples). The autogenous-shrinkage
  *valley* depths of manuscript Table 3 are now computed from it by `Src/t13_strain_valley.py`
  instead of being quoted from an unreproducible export. Under the data-release agreement this
  record is supplied **on request** rather than deposited — see *Data release scope* below.
- **W/C 0.50 3-day strengths corrected** to the strength-test report (air 18.9, water 19.3;
  previously 19.3 / 20.4) and W/C 0.60 air-dried 28-day to 14.9. This is the only strength
  correction; the other 30 cube values were unchanged.
- **Sensor-free hyperbolic refit is now persisted** (`Src/t2b_sensorfree_refit.py` →
  `Data/hyperbolic_fits_sensorfree.csv`). The manuscript quotes this refit; it previously
  existed only as an ad-hoc calculation.
- **SI tables S1–S3 and S6 are now generated** from `Data/` by `Src/make_si_tables.py`, and
  the member-size sweep writes `Data/vs_sweep_050.csv`. Both were hand-assembled before, so
  pipeline corrections did not propagate into the submitted package.
- Every quantitative claim in the manuscript was cross-checked against this package; the
  audit and its outcomes are recorded in `Doc/verification_notes_r11.md`.

Carried over from r10.2: cube core maturities are integrated from the raw 2-min
thermocouple logs (indoor members = channels 1–3 mean; outdoor member = channel 4); the
member-size sweep includes the regularised L = 800 mm case.

**Temperature provenance.** All temperatures in this package and in the manuscript are the
original **core-thermocouple** records. FBG data enter only as **strain**.

## Data release scope

Three categories, set by the data owners. The pipeline is built around this split: every
stage that needs a request-only record skips cleanly when it is absent, so the public
package runs end to end as downloaded.

| Category | Content | Release |
|---|---|---|
| **Literature compilation** | `lit503`, `lit119` and its `lit55` subset inside `SI_Table_S1`; `SI_Table_S2`, `SI_Table_S3` | **Open.** Re-tabulated from published sources with per-row citation (`ref`) and per-study provenance (`SI_sources.csv`). Cite the original studies as well. |
| **Cube thermocouple temperature** | `SI_Table_S7` (10 098 samples, 12 core channels + ambient T/RH) and everything derived from it | **Open, permanent.** |
| **Cube FBG strain** | 2-min embedded autogenous-strain series (30 700 samples) | **On request** to the corresponding author. Not deposited. |
| **Field-slab sensing logs** | Raw FBG/IoT logs of the two reference slabs | **On request**, under the access terms of the prior publication (*Front. Mater.* **13**:1762995). The slab strength and maturity values reported in that paper are re-tabulated here with citation. |

Derived quantities from the request-only records — the Table 3 valley depths, the slab
maturity of `SI_Table_S4` — are reported in the article and in this deposit; only the
underlying time series are withheld. `Src/check_reproduction.py` marks the four valley
checks **SKIP** rather than FAIL when the strain record is absent, so a reader sees
**39/39** where the authors see 43/43.

## Repository contents (public)

This repository is the public deposit and contains everything needed to reproduce the
published results:

- **`Supplementary_Dataset/`** — the reference literature dataset for the machine-learning
  analysis and the raw experimental records of the case member (core-thermocouple
  temperature, embedded-FBG strain), plus the derived tables.
- **`Src/`** — the complete analysis pipeline, MIT-licensed, with a single entry point
  (`Src/run_all.py`) and a pinned environment (`pyproject.toml`, `uv.lock`, at the root).

Not deposited: `Data/` (intermediate products, created by the run) and `Doc/` (manuscript
sources, figures and internal review records). The manuscript itself is distributed by the
journal.

The dataset is the **complete, cleaned collection** used to compute concrete maturity
indices and to train/evaluate the compressive-strength–prediction models. All strength
records are paired with their curing temperature history reduced to three standard
maturity indices (Nurse–Saul, Arrhenius equivalent age, degree-day). Every table is
plain UTF-8 CSV (with BOM, Excel-friendly) and is self-describing through
`SI_data_dictionary.csv`.

## Requirements

| | |
|---|---|
| **Python** | 3.11–3.13 (`requires-python = ">=3.11,<3.14"`). Results were produced on **3.12.13**. |
| **Package manager** | [**uv**](https://docs.astral.sh/uv/) ≥ 0.5 — a single static binary, no pre-existing Python needed. Results were produced with **0.11.28**. |
| **Environment** | Declared in `pyproject.toml` at the repository root, pinned in `uv.lock` (36 resolved packages). `uv run` materialises it on first use; nothing to install by hand. |
| **Disk / time** | ~1 GB for the environment, ~5 MB for outputs. A full run takes 3–6 min on a laptop. |
| **OS** | Platform-independent. No absolute paths; the only optional external input (ABAQUS output) is located through the `CONCRETE_ABAQUS_DIR` environment variable. |

Installing uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                      # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"           # Windows
```

**Direct dependencies** (declared) and the versions the published results were produced
with (resolved):

| Package | Declared | Used | Role |
|---|---|---|---|
| `pandas` | ≥ 2.2 | 3.0.3 | table handling, CSV/parquet I/O |
| `numpy` | ≥ 1.26 | 2.4.6 | numerics |
| `scipy` | ≥ 1.11 | 1.18.0 | curve fitting (hyperbolic fits), χ² intervals |
| `scikit-learn` | ≥ 1.4 | 1.9.0 | the six baseline learners, grouped cross-validation |
| `xgboost` | ≥ 2.0 | 3.3.0 | monotonic-constraint hybrid (H2) |
| `shap` | ≥ 0.46 | 0.52.0 | feature attribution (Fig. 11) |
| `numba` | ≥ 0.59 | 0.66.0 | `shap` dependency, pinned away from 2021-era builds |
| `matplotlib` | ≥ 3.8 | 3.11.0 | figures |
| `pyarrow` | ≥ 15 | 24.0.0 | parquet engine |
| `openpyxl` | ≥ 3.1 | 3.1.5 | reads the raw `.xlsx` sources (authors' full mode only) |

## How to run

```bash
uv run python Src/run_all.py              # rebuild everything          (3-6 min)
uv run python Src/check_reproduction.py   # assert the published numbers
```

Run from the repository root, or from `Src/` with the paths dropped — `uv run` finds the
project either way.

The second command prints a pass/fail line per quantity and ends with

```
  39/39 checks passed.
  This run reproduces every published number that Data/ can supply.
```

Anything else means a genuine discrepancy — the table names the quantity, the value obtained
and the value expected. Please report it with that line.

Individual stages can be run on their own, in the order listed under
the [manifest](#file-manifest); each reads what the previous ones wrote to `Data/`.

**Two modes, detected automatically.** With this deposit you get *deposit mode*:
`t0_bootstrap_from_si.py` reconstructs `Data/` from the deposited tables, then every compute
stage runs unchanged. On the authors' machine, where the raw spreadsheets are present, *full
mode* parses them from scratch instead. Both produce identical numbers; set
`CONCRETE_RUN_MODE=deposit` to force the reader's path. Two of the raw inputs cannot be
redistributed — the literature workbook is a third-party compilation and the field-slab logs
belong to the prior publication — which is why the deposit ships their derived form.

`REPRODUCE.md` has the step-by-step walkthrough, the mapping from each manuscript table and
figure to the script that produces it, and the one result that needs a solver (the
finite-element member-size sweep of Table 7; its output table is deposited and checked, but
re-running the solve requires ABAQUS).

## File manifest

Every file in the deposit, one line each.

**Root**

| File | Description |
|---|---|
| `README.md` | This file — dataset composition, provenance, conventions, manifest, change log. |
| `REPRODUCE.md` | Reproduction walkthrough: requirements, commands, result-to-script mapping. |
| `LICENSE` | Which licence applies to what, plus the citation request. |
| `LICENSE-MIT.txt` | MIT text, governing `Src/`. |
| `LICENSE-CC-BY-4.0.txt` | CC BY 4.0 text, governing the data and documentation. |
| `pyproject.toml` | Project metadata, the Python floor and ceiling, and the direct dependencies. At the root so `uv run` resolves from anywhere in the tree. |
| `uv.lock` | Fully resolved dependency graph — the exact versions behind the published numbers. |
| `.gitignore` | Excludes caches, the local environment, and internal working documents. |

**`Supplementary_Dataset/` — the data** (2.2 MB; full column definitions in `SI_data_dictionary.csv`)

| File | Rows | Description |
|---|---:|---|
| `SI_Table_S1_strength_maturity.csv` | 732 | **Primary dataset.** One row per strength test: mix design, curing condition, the three maturity indices, measured strength. Written in pipeline row order at full precision so the reconstruction is exact. |
| `SI_Table_S2_hyperbolic_fits.csv` | 105 | Per-mixture ASTM C1074 hyperbolic fit (`Su`, `k`, `R²`, RMSE, n). |
| `SI_Table_S3_activation_energy.csv` | 10 | Per-mixture apparent activation energy from the Arrhenius calibration. |
| `SI_Table_S4_core_vs_ambient_maturity.csv` | 40 | Core- versus ambient-history maturity of the instrumented members, with the ambient→core transfer factor. |
| `SI_Table_S5_member_size_sweep.csv` | 11 | Calibrated finite-element member-size sweep, L = 100–800 mm × W/C 0.50 / 0.60. |
| `SI_Table_S7_core_temperature.csv` | 10 098 | **Raw core-thermocouple log**, twelve core channels plus ambient temperature and humidity, 2-min over 14 d, one shared clock. |
| `SI_data_dictionary.csv` | 59 | Codebook: every column of every table with unit, type and description. |
| `SI_sources.csv` | 33 | Per-source provenance: study, subset, record and mix counts, role, raw file. |
| `surrogate_lookup.csv` | 8 | Screening lookup of `Su` and `k` against W/C (Eq. for the tier-1 prior). |
| `SI_README.md` | — | Short in-folder pointer for the journal submission copy. |

**`Src/` — the pipeline** (MIT)

*Configuration and shared code*

| File | Description |
|---|---|
| `config.py` | Paths, physical constants, maturity conventions, and the run-mode detection. |
| `maturity.py` | Maturity mathematics: Nurse–Saul, Arrhenius equivalent age, degree-day, hydration degree. |
| `ml_utils.py` | Feature groups, the six-model factory, metrics, and grouped cross-validation helpers. |
| `physics_prior.py` | The hyperbolic strength–maturity prior and the pooled `Su`(W/C) fallback. |

*Pipeline stages, in run order*

| File | Description |
|---|---|
| `run_all.py` | **Entry point.** Detects the run mode and executes every stage in order. |
| `t0_bootstrap_from_si.py` | *Deposit mode ingest:* rebuilds `Data/` from the deposited tables. |
| `t8_core_logs.py` | *Full mode ingest:* integrates maturity from the raw 2-min temperature logs. |
| `t1_build_master.py` | *Full mode ingest:* parses and harmonises the sources into the 732-record pool. |
| `t2_hyperbolic.py` | Fits the per-mix hyperbolic law and attaches the physics prior to every record. |
| `t2b_sensorfree_refit.py` | Refits the cube excluding the sensor-bearing 28-day companions. |
| `t3t4_hybrid_cv.py` | Leakage-controlled cross-validation: pure learners versus the physics hybrids. |
| `t10_hpo.py` | Hyper-parameter search for the winning hybrid under grouped cross-validation. |
| `t5t6_slab.py` | Transfer to the two field slabs (leave-experiment-out). |
| `t11_tempweighted.py` | Temperature-weighted maturity variant and its sensitivity. |
| `t9_calibration.py` | ASTM C1074 in-place calibration from a single early core. |
| `t7_uncertainty_decision.py` | Conformal prediction intervals and the code-referenced striking decision. |
| `t12_shap.py` | SHAP attribution of the hybrid model. |
| `t13_strain_valley.py` | Autogenous-shrinkage valley depths from the FBG strain record (skips if the request-only record is absent). |
| `make_figures.py` | Regenerates the manuscript figures into `Doc/figures/`. |
| `make_deliverables.py` | Consolidates every experiment into `Data/results_table.csv`. |
| `make_si_tables.py` | Regenerates the deposited tables S1–S3 and S7 from `Data/`; writes the request-only strain table to `Restricted/`. |

*Verification and finite-element post-processing*

| File | Description |
|---|---|
| `check_reproduction.py` | Asserts the published quantities against this run (43 with the request-only record, 39 without); exits non-zero on any mismatch. |
| `vs_sweep_postproc.py` | Turns ABAQUS core-temperature output into the W/C 0.50 member-size sweep. |
| `vs_sweep_compare_060.py` | Same for W/C 0.60, isolating the moisture contribution from the thermal one. |

**`Data/` — created by the run**, not part of the download: the harmonised pool
(`master.parquet`, `master_phys.parquet`), the per-subset CSVs, and one file per result
(`hyperbolic_fits*.csv`, `t3t4_*.csv`, `hpo_*.csv`, `slab_validation.csv`,
`calibration_results.csv`, `decision_exceedance.csv`, `shap_importance.csv`,
`strain_valley.csv`, `vs_sweep_*.csv`, `results_table.csv`, …).

## The deposited tables in detail

The one-line summaries are in the [manifest](#file-manifest); this section adds the
provenance and reading caveats a re-user needs.

| File | Rows | Content |
|---|---:|---|
| `SI_Table_S1_strength_maturity.csv` | 732 | **Primary dataset** — one row per strength test: mix design, curing condition, three maturity indices, and measured strength `f_c` (the model target). |
| `SI_Table_S2_hyperbolic_fits.csv` | 105 | Per-mixture ASTM C1074 hyperbolic fit (ultimate strength `Su`, rate `k`, `R²`, RMSE). |
| `SI_Table_S3_activation_energy.csv` | 10 | Per-mixture apparent activation energy `Eₐ` from ASTM C1074 Arrhenius calibration. |
| `SI_Table_S4_core_vs_ambient_maturity.csv` | 40 | Core- vs ambient-history maturity for the field members (ambient→core transfer factor); cube rows split into the indoor-member mean (`cube`) and the field-exposed member (`cube_out`). |
| `SI_Table_S5_member_size_sweep.csv` | 11 | Calibrated finite-element member-size sweep (cube least dimension `L`=100–800 mm × W/C 0.50/0.60; the `L`=800 mm row is the heat-transfer-only solve with the regularised hydration rate): peak and mean-first-72 h core temperature, core-to-standard maturity ratio (the assumption-free FE output), temperature-dependent core `Su`, and in-place/standard strength ratio at 2 d / 7 d / ~14 d (model-dependent). The core-temperature field is mix-independent by construction, so the maturity ratio is identical across W/C. `Su` is refit from the sensor-free strength data (excluding the FBG-compaction-affected 28-day companions). Generated by `Src/vs_sweep_postproc.py` and `Src/vs_sweep_compare_060.py` from the ABAQUS runs in `2026_Vs_sweep/` and `2026_Vs_sweep_060/`. |
| `SI_Table_S7_core_temperature.csv` | 10 098 | **Raw core-thermocouple log** of the 400 mm cube: twelve core channels (`TC-<mix>-1..4`; members 1–3 indoor, member 4 field-exposed) plus indoor/outdoor ambient temperature and relative humidity, 2-min sampling over 14 d. These are the original records and are the **only** temperature source used anywhere in this work. Wide form with a single shared clock — the logger ran continuously from before the first mix was placed, so the channels are row-aligned. The mix-mean peaks of manuscript Table 3 follow directly: mean of the four member peaks = 41.02 / 42.48 / 38.34 °C for A / B / C. |
| `SI_sources.csv` | 33 | Provenance table: every source study, its S1 subset, record/mix counts, role, and raw file. |
| `SI_data_dictionary.csv` | — | Codebook: every column with unit, type and description. |
| `SI_README.md` | — | Short in-folder pointer (journal submission copy). |

**What this package publishes.** Two bodies of data, both complete:

1. **The reference literature dataset** used for the machine-learning analysis — `lit503`,
   `lit119` and its `lit55` subset inside `SI_Table_S1`, with full per-study provenance in
   `SI_sources.csv`.
2. **The raw core-thermocouple record of the case member** (`SI_Table_S7`), the temperature
   source for every result reported. The companion FBG strain record is released on request
   (see *Data release scope*).

The remaining tables (S2–S5) are derived quantities kept for convenience; they are
reproducible from 1 and 2 via `Src/run_all.py`.

The compressive-strength report behind the replicate means and their coefficients of
variation (0.1–10 %) is reproduced as **Fig. S1** of `Doc/Supplementary_Figures_CSCM.pdf`
rather than transcribed to CSV: each bar there carries `mean ± CoV`, and several labels
overlap, so re-typing them would risk transcription error. The values tabulated in
`SI_Table_S1` are the means shown in that figure.

## Primary dataset composition (S1)

| Subset | Rows | Mixes | Sources | f_c range (MPa) | W/C range | Curing T (°C) |
|---|---:|---:|---:|---|---|---|
| `lit55`  | 55  | 5   | 1  | 3.6 – 36.0  | 0.41 – 0.55 | 5 – 40 |
| `lit119` | 119 | 10  | 1  | 3.6 – 60.0  | 0.32 – 0.83 | 5 – 80 |
| `lit503` | 503 | 169 | 32 | 2.1 – 101.0 | 0.23 – 0.80 | 0 – 40 |
| `slab`   | 22  | 2   | 1  | 10.9 – 36.2 | 0.44 – 0.64 | 16 – 28 |
| `cube`   | 33  | 3   | 1  | 3.5 – 31.3  | 0.50 – 0.60 | 20 – 41 |
| **Total** | **732** | **184** | **35** | 2.1 – 101.0 | 0.23 – 0.83 | 0 – 80 |

`lit55` is the reference study's exact 55-point identified subset and is nested inside
`lit119`; both derive from Ryu et al. (2024). `lit503` is the large curated literature
database. `slab` and `cube` are the in-situ experiments used for out-of-domain
(field-transfer) evaluation.

## Data provenance

Full per-source provenance is in `SI_sources.csv`. Summary:

| Data origin | Records | Mixes | Sources | Role |
|---|---:|---:|---:|---|
| Literature (compiled) | 677 | 179 | 30 studies | Model training and leakage-free cross-validation |
| Author's experiments | 55 | 5 | 2 (slab, cube) | Out-of-domain field validation |

- **Literature** — all literature records were compiled into a single collection spreadsheet
  (`2025_1127_Collected Data maturity.xlsx`) in the parent project directory. `lit55`/`lit119`
  come from **Ryu et al. (2024)**; `lit503` pools **~30 Korean Concrete Institute–era studies**
  (Kim, Han, Lee, Park, Choi, Kwon, Nam, Kang, Khil, Lim, Seo, Yoo, KCI 2004, …).
- **Author's experiments (`slab`, `cube`)** — parsed from the raw field/laboratory records in
  the parent directory: `SLAB_Exp_1.xlsx`, `SLAB_Exp_2/` (folders `A`, `C`, `IoT`) and the
  slab paper `2025_slab_fmats-13-1762995.pdf` (Table 3) for the slabs; for the cube,
  `CUBE_TC-Mockup.xlsx` (12 core thermocouples + indoor/outdoor ambient, 2-min, 14 d),
  `CUBE_FBG-Strain-embedded.xlsx` and `CUBE_FBG-Strain-surface.xlsx` (FBG strain; the
  surface gauges were installed one day after casting and so do not see the early valley),
  `2026_0611_cube_report_final_a.pdf` and `2026_0611_압축강도_시험결과_rik.png`.

  *Excluded by decision of the authors:* a separate set of strain traces (Chosun University
  interrogator, `D:\2026_hetval\심부센서사진\조선대_*.png`) is **erroneous measurement and is
  not used in this work.** Its sign and magnitude are opposite to
  `CUBE_FBG-Strain-embedded.xlsx`, so it is flagged here to prevent confusion.

**Reference-string normalisation.** Source labels in the raw literature spreadsheet were
inconsistently formatted (e.g. `Han et al.,2000` / `Han et al., 2000`; `kim et al., 1998`;
`Kim et l., 1996`; `Yoo et. al., 2017`). These were normalised (whitespace, `et al.`
spelling, leading capitalisation) — collapsing purely typographic duplicates from 32 raw
label strings to 30 distinct studies — **without merging different publication years**. The
normalised label is used consistently in `SI_Table_S1`, `SI_Table_S2` and `SI_sources.csv`.
Derived tables carry a `ref` column so every record traces back to its source.

## Maturity conventions

All temperature histories were placed on one convention (ASTM C1074):

- **Nurse–Saul** `M_NS = Σ (T − T₀) Δt`, datum `T₀ = −10 °C` — units °C·h.
- **Arrhenius equivalent age** `t_eq = Σ exp[−Eₐ/R (1/T − 1/T_ref)] Δt` at
  `T_ref = 20 °C = 293.15 K`, default `Eₐ = 38 300 J/mol` — units h.
- **Degree-day** `DD = Σ T Δt / 24` — units °C·day.

Note: `T_ref = 293.15 K` (physically correct for 20 °C) is used, correcting the
"298.15 K (20 °C)" printed in the source paper (298.15 K is 25 °C). `Eₐ` is additionally
calibrated per binder where the data allow (Table S3); its effect on `t_eq` is negligible
near 20 °C and becomes material only for the hot-cured field members.

## Data-cleaning notes (for transparency)

The tables here are harmonised from heterogeneous source spreadsheets. The following
deliberate decisions were applied and are documented so results are fully reproducible:

- **Column harmonisation.** Fly-ash and slag/SCM contents from the literature subsets are
  unified into `fly_ash` and `SCM_other`; `binder = cement + fly_ash + SCM_other`.
- **Blanks are genuine missing values**, not zeros. `cement`/`binder` are blank for the
  `slab`/`cube` mixes because those proprietary mix designs are not disclosed; `curing_type`
  is blank for the isothermal literature (not applicable).
- **Dropped columns.** A corrupted water–binder-ratio column (inconsistent units across
  sources — e.g. reported as 42–50 in one subset and 0.24–0.80 in another) and an empty
  stray column were removed; `WC` (reliable in every subset) is retained as the mix-design
  parameter. Model-engineered features (`√DD`, `log t`, hydration degree α, hyperbolic
  prior, etc.) are **not** included here — they are deterministically reproducible from these
  raw quantities via the published pipeline.
- **Cube effective W/C.** The 400 mm cube mixes are reported at their *effective* W/C
  (≈0.50–0.60), consistent with the strength/heat back-analysis in the manuscript, not the
  nominal batch W/C.
- **Rounding.** Numeric fields are rounded to physically meaningful precision (ratios 3 dp,
  temperatures/strengths 1–2 dp, maturities 1–2 dp). No records were removed or imputed.

**Notes (round 10.2).** The cube source label in S1/S2 is "This study (cube)". In
`t3t4_bigpool_results.csv` the leave-mix-family-out pool holds n=502 of the 503 records: one
record lacks the features required by the learner after harmonisation and is dropped by the
pipeline. `shap_importance.csv` and Fig. 11 were regenerated on the corrected master dataset
(`t12_shap.py`, added to the pipeline): mean |SHAP| of S_hyp = 12.90 MPa (was 13.12 before the
core-maturity correction); the prior-dominance conclusion is unchanged.

**Notes (round 11.0) — corrections and what they move.** Every quantitative claim in the
manuscript was recomputed from this package (audit trail: `Doc/verification_notes_r11.md`).
Two data corrections resulted.

*W/C 0.50, 3-day strengths.* Against the strength-test report (labelled *after outlier
removal*), 31 of the 33 cube strengths matched exactly. Two did not: the air-dried 3-day
value in the pipeline equalled the report's *water-cured* 3-day value, and the pipeline's
water-cured 3-day value (20.4 MPa) appears nowhere in the report — a one-cell shift. Both
are corrected to the report (air 18.9, water 19.3), together with W/C 0.60 air-dried 28-day
(15.0 → 14.9). Effect, after re-running the pipeline:

| Quantity | r10.2 | r11.0 |
|---|---|---|
| W/C 0.50 sensor-free `Su`, `k` | 27.27 MPa, 0.0229 | **27.43 MPa, 0.0212** |
| Size sweep, in-place/standard @ ~14 d (L = 100→800 mm) | 0.97 / 0.96 / 0.94 / **0.91** / 0.86 / 0.86 | **unchanged** |
| Size sweep @ 2 d (L = 100→800 mm) | 1.08 → 1.28 | 1.09 → 1.31 |
| Core `Su` (L = 100→600 mm, 800 mm) | 26.1 → 22.6, 22.2 | 26.3 → 22.7, 22.4 |
| W/C 0.555 and 0.60 fits | 30.59 / 23.49 | **unchanged** |

The strength ratio uses the same `k` in numerator and denominator, so the late-age column —
and with it the paper's headline 0.91 member-size result — is insensitive to the change.

*Autogenous valley depths.* Recomputed from the newly included strain record as the minimum
of the embedded channel over the first 3 days. Mix A reproduces the previously published
value exactly (−149.7 µε). Mix B is now −292.8 µε: the previous −299.2 µε is deeper than any
sample in the 14-day record and cannot be obtained from it under any window or smoothing
choice, so it came from a different processing of the raw wavelengths. Mix C is now −40.8 µε
(previously −37.8 µε, which corresponds to the local plateau rather than the minimum — the
raw minimum sits on a single noise spike 2.8 µε below it; `strain_valley.csv` reports both
the raw minimum and a 30-min median for this reason). The W/C linearity that the manuscript
relies on is unaffected (R² = 0.9996).

*Terminology.* The manuscript previously used *valley* (11×, including the FE model's formal
parameters `V_A`, `τ_v`, `K_vpk`) and *trough* (9×) for the same feature; all are now
**valley**.

*Not resolved.* The `load_cube()` docstring in `t8_core_logs.py` quotes outdoor-member
ambient correlations of ~0.45–0.49 against ~0.2 for the indoor members; the measured values
are 0.16–0.30 over the full record and 0.34–0.39 after 3 days. This figure is not cited in
the manuscript, and the indoor/outdoor split is independently justified by the casting
temperatures (29–31 °C outdoors vs ~24 °C indoors) and peaks (46.3 °C), so the comment is
wrong but nothing downstream depends on it.

## Directory layout

```
CVS/
├── README.md, REPRODUCE.md, LICENSE*          this documentation
├── pyproject.toml, uv.lock                    the pinned environment (uv project root)
├── Supplementary_Dataset/                     input  — the deposited tables (S1-S5, S7 + metadata)
├── Restricted/                                request-only records; never deposited
├── Src/                                       the pipeline (uv project; run_all.py)
├── Data/                                      output — created by the run, not downloaded
└── Doc/                                       not deposited — manuscript sources and figures
```

In the authors' tree the raw inputs (`*.xlsx`, `*.pdf`, `*.png`, the `SLAB_Exp_*` logs) sit in
the parent directory and are located through `config.py:ROOT_DIR`; their absence is what
selects deposit mode, so nothing needs editing. The two finite-element post-processing scripts
are the only ones that read outside the tree — point `CONCRETE_ABAQUS_DIR` at your ABAQUS runs
if you want to regenerate Table 7 from the solver rather than use the deposited table.

## License

| Content | License |
|---|---|
| Analysis code (`Src/`) | MIT — see `LICENSE-MIT.txt` |
| Data, documentation, figures (`Supplementary_Dataset/`, `Data/`, READMEs) | CC BY 4.0 — see `LICENSE-CC-BY-4.0.txt` |

## Citation

If you use this dataset or code, please cite the accompanying article (details in
`LICENSE`) and, for the underlying slab framework:

> Park, Choe, Kim & Rhee (2026). *Frontiers in Materials* **13**:1762995.

Literature-derived records remain the property of their respective authors and preserve
their source attributions (`SI_sources.csv`, `ref` column); the cube and slab
measurements are original data of the authors.
