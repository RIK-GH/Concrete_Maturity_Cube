"""
config.py - Central configuration and physical constants.

Advancing Park, Choe, Kim & Rhee (Front. Mater. 13:1762995, 2026):
maturity-strength framework for slab formwork-removal decisions.

Conventions (Korean Windows 11):
  * SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  -> all paths anchored here
  * CSV I/O encoding: the user pattern is encoding='latin-1'. However, the source
    database contains Korean reference names, en-dashes and unicode superscripts
    (e.g. 'Maturity (°C·t^0.3)', 'W/C 0.30 - 10 °C') that are NOT representable in
    latin-1 and would raise UnicodeEncodeError / corrupt text. We therefore write
    CSVs as 'utf-8-sig' (Excel-friendly BOM) and READ legacy latin-1 files with a
    latin-1->utf-8 fallback. This is a deliberate, documented override for data
    integrity. Binary parquet is unaffected.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CVS_DIR = os.path.dirname(SCRIPT_DIR)                    # .../CVS
ROOT_DIR = os.path.dirname(CVS_DIR)                      # the data-collection folder (raw sources)
FIG_DIR = os.path.join(CVS_DIR, "Doc", "figures")        # figures consumed by the manuscript
OUT_DIR = os.path.join(CVS_DIR, "Data")                  # CSV/parquet consumed by the manuscript
SI_DIR = os.path.join(CVS_DIR, "Supplementary_Dataset")  # the public deposit
# Request-only material. Under the data-release agreement the FBG strain record is supplied
# to requesters rather than deposited, so it lives here and is excluded from version control.
# Absent for anyone who downloaded the public package - every stage that reads it must skip
# cleanly rather than fail.
RESTRICTED_DIR = os.path.join(CVS_DIR, "Restricted")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# TWO RUN MODES
#
# 'full'    - the authors' tree, with the raw source files sitting in ROOT_DIR.
#             The pipeline parses them from scratch (t8 -> t1 -> ...).
# 'deposit' - what a reader who downloads the public package gets: Src/ and
#             Supplementary_Dataset/ only. The raw spreadsheets are not part of the
#             deposit (the literature workbook is a third-party compilation and the
#             field-slab logs belong to the prior publication), but everything the
#             manuscript reports is derivable from the deposited tables, so
#             t0_bootstrap_from_si.py rebuilds Data/ from them and the rest of the
#             pipeline runs unchanged.
#
# Nothing needs to be configured: RUN_MODE is detected below, and run_all.py picks
# the right stage list. Set CONCRETE_RUN_MODE=deposit to force the reader's path
# even when the raw files are present (useful for checking the deposit is complete).
# --------------------------------------------------------------------------
def _raw_present():
    probe = ["2025_1127_Collected Data maturity.xlsx", "CUBE_TC-Mockup.xlsx"]
    return all(os.path.exists(os.path.join(ROOT_DIR, p)) for p in probe)


RUN_MODE = os.environ.get("CONCRETE_RUN_MODE") or ("full" if _raw_present() else "deposit")
HAVE_RAW = RUN_MODE == "full"

# ABAQUS member-size sweep output. Only needed to regenerate Table 7 / SI Table S5 from
# the solver; the resulting table is deposited, so this is optional. Override with
# CONCRETE_ABAQUS_DIR when the runs live elsewhere.
ABAQUS_DIR = os.environ.get("CONCRETE_ABAQUS_DIR", r"D:/2026_hetval")

CSV_ENC = "utf-8-sig"     # write; see module docstring for rationale

# ---- source files (parsed, not assumed) ----------------------------------
XLSX = os.path.join(ROOT_DIR, "2025_1127_Collected Data maturity.xlsx")
SLAB_PDF = os.path.join(ROOT_DIR, "2025_slab_fmats-13-1762995.pdf")
CUBE_PDF = os.path.join(ROOT_DIR, "2026_0611_cube_report_final_a.pdf")
CUBE_PNG = os.path.join(ROOT_DIR, "2026_0611_압축강도_시험결과_rik.png")

# Cube core-thermocouple log (2-min, 14 d): 12 core channels (TC-<mix>-1..4; members 1-3
# indoor, member 4 field-exposed) plus indoor/outdoor ambient temperature and humidity.
# These are the only temperature source used anywhere in this work.
CUBE_TC = os.path.join(ROOT_DIR, "CUBE_TC-Mockup.xlsx")

# Cube FBG strain records (2-min, 14 d). Temperature-compensated: the gauges report strain
# with the thermal component already removed (one strain + one temperature-compensation FBG
# per mix), so the embedded channel is the autogenous+drying strain the UMAT is compared to.
# Strain zero is set 1-2 h BEFORE completion of placement of that mix, so each mix's series
# starts at its own casting time (A, then B ~1.7 h later, then C ~3.0 h later).
CUBE_STRAIN_EMB = os.path.join(ROOT_DIR, "CUBE_FBG-Strain-embedded.xlsx")   # A-I/B-I/C-I core
CUBE_STRAIN_SUR = os.path.join(ROOT_DIR, "CUBE_FBG-Strain-surface.xlsx")    # A-O/B-O/C-O surface

# ==========================================================================
# PHYSICAL CONSTANTS  (maturity theory)
# ==========================================================================
R_GAS = 8.314                 # J/mol/K, universal gas constant

# Nurse-Saul datum temperature (ASTM C1074 default; paper uses -10 C)
T0_NS = -10.0                 # deg C

# Arrhenius reference temperature.
#   NOTE ON A PAPER INCONSISTENCY: the source paper text states
#   "Tref = 298.15 K (20 °C)". 298.15 K is 25 °C, NOT 20 °C. The correct
#   absolute temperature for a 20 °C reference is 293.15 K. We use the
#   physically-correct 293.15 K and expose TREF_C so sensitivity to the
#   paper's 298.15 K value can be quantified.
TREF_C = 20.0                 # deg C
TREF_K = TREF_C + 273.15      # = 293.15 K  (physically correct)
TREF_K_PAPER = 298.15         # the (erroneous) value printed in the paper

# Apparent activation energy for equivalent age (Type I cement default).
# Paper fixes EA = 38,300 J/mol. We CALIBRATE per-binder via ASTM C1074 Annex
# (Arrhenius plot of rate constant vs 1/T) and report sensitivity vs this value.
EA_DEFAULT = 38300.0          # J/mol

# Bazant / Freiesleben-Hansen hydration-degree parameters (paper: alpha_inf=0.88,
# beta=0.85). Two functional forms are supported in maturity.py:
#   (Bazant, paper Eq.13)  alpha(M) = alpha_inf * (M/(tau+M))**beta
#   (Freiesleben-Hansen)   alpha(te) = alpha_inf * exp(-(tau/te)**beta)
ALPHA_INF = 0.88
BETA_HYD = 0.85
TAU_HYD_DAYS = 0.30           # provisional hydration time parameter (days); calibrated later

# ==========================================================================
# CODE-BASED FORMWORK / SHORE REMOVAL THRESHOLDS  (decision layer, T7)
# ==========================================================================
# KCS 14 20 12 (Korea): formwork strike when S >= max(14 MPa, 2/3 f'c)
# ACI 318 / 347R      : ~70% f'c for formwork; 100% f'c for shores
# EN 13670            : 2-5 MPa or 50-70% f'c
CODE_THRESHOLDS = {
    "KCS_absolute_MPa": 14.0,
    "KCS_frac_fck": 2.0 / 3.0,
    "ACI_frac_fck": 0.70,
    "ACI_shore_frac_fck": 1.00,
    "EN_absolute_MPa": 5.0,
    "EN_frac_fck": 0.60,
}

# reproducibility
RANDOM_STATE = 42
