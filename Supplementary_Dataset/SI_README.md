# Supplementary Dataset — Concrete maturity and compressive-strength prediction

Supporting dataset for the manuscript **"Member-size dependence of in-place concrete
strength: sensor-monitored maturity for formwork removal"** (Park, Choe, Kim & Rhee;
under review, *Case Studies in Construction Materials*). All temperatures in this package
are original core-thermocouple records; strain quantities derive from the FBG
gauges.

**Package version: r11.0 (2026-07-27).**

**Release scope.** The literature compilation (with per-study citation) and the cube
core-thermocouple record are openly deposited. Two records are released by the data owners
**on reasonable request** instead: the cube's embedded-FBG autogenous-strain series and the
raw sensing logs of the two reference field slabs (the latter under the access terms of
*Front. Mater.* **13**:1762995). Quantities derived from them are reported in the article and
in this package; the analysis code is included, so the derivation is inspectable.

## Files

| File | Rows | Content |
|---|---:|---|
| `SI_Table_S1_strength_maturity.csv` | 732 | **Primary dataset** — one row per strength test: mix design, curing condition, three maturity indices (Nurse–Saul, Arrhenius equivalent age, degree-day), measured strength `f_c`. |
| `SI_Table_S2_hyperbolic_fits.csv` | 105 | Per-mixture ASTM C1074 hyperbolic fits (`Su`, `k`, `R²`, RMSE). |
| `SI_Table_S3_activation_energy.csv` | 10 | Per-mixture apparent activation energy `Eₐ`. |
| `SI_Table_S4_core_vs_ambient_maturity.csv` | 40 | Core- vs ambient-history maturity (field members; ambient→core transfer factor). |
| `SI_Table_S5_member_size_sweep.csv` | 11 | Calibrated FE member-size sweep (L = 100–800 mm × W/C 0.50/0.60). |
| `SI_Table_S7_core_temperature.csv` | 10 098 | **Raw core-thermocouple log** of the 400 mm cube: twelve core channels (`TC-<mix>-1..4`; members 1–3 indoor, member 4 field-exposed) plus indoor/outdoor ambient temperature and RH, 2-min over 14 d. Original records; the only temperature source used in this work. One shared clock, so the channels are row-aligned. Table 3 mix-mean peaks = mean of the four member peaks: 41.02 / 42.48 / 38.34 °C for A / B / C. |
| `SI_sources.csv` | 33 | Per-source provenance (study, subset, counts, role, raw file). |
| `SI_data_dictionary.csv` | — | Codebook: every column with unit, type, description. |

**Full documentation** — dataset composition, provenance and reference normalisation,
maturity conventions, and data-cleaning notes — is in the repository root `README.md`
(https://github.com/RIK-GH/Concrete_Maturity_Cube).

## License and citation

Data are licensed CC BY 4.0 (repository `LICENSE-CC-BY-4.0.txt`); analysis code MIT.
If you use this dataset, please cite the accompanying article and, for the underlying
slab framework, Park, Choe, Kim & Rhee (2026), *Frontiers in Materials* **13**:1762995.
Literature-derived records remain the property of their respective authors (see the
`ref` column and `SI_sources.csv`); the cube and slab measurements are original data
of the authors.
