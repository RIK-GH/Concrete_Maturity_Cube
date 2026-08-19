"""
vs_sweep_compare_060.py  --  W/C 0.50 vs 0.60 sweep: test thermal-field mix-independence and
isolate the moisture (self-desiccation-relief) contribution to the high-W/C surplus (Reviewer M4).

Logic:
  * heat generation / conductivity / specific heat are IDENTICAL across mixes (only WCRATIO &
    D_skin differ) -> the temperature field, hence the maturity ratio, should be mix-independent.
  * if so, the measured 0.60 SURPLUS (1.17-1.27) vs 0.50 DEFICIT (0.82-0.99) cannot be thermal;
    the difference is the moisture mechanism the FE temperature field cannot produce.
"""
import numpy as np, os
from config import ABAQUS_DIR
Ea, R, Tref_K = 38300.0, 8.314, 293.15
S050 = os.path.join(ABAQUS_DIR, "2026_Vs_sweep")
S060 = os.path.join(ABAQUS_DIR, "2026_Vs_sweep_060")
SIZES = [100, 200, 300, 400, 600]     # 400 mm = actual specimen size (validation anchor)

# temperature-dependent ULTIMATE strength shape (crossover 30/27/22 @ 5/20/40 C, from the paper);
# use the fractional shape Su(T)/Su(20) so it can be anchored to each mix's own Su_std.
CT = [5.0, 20.0, 40.0]; CS = [30.0, 27.0, 22.0]
def su_shape(Tc): return float(np.interp(Tc, CT, CS)) / 27.0
SU_STD = {"0.50": 27.428, "0.60": 22.230}     # cube hyperbolic Su per mix (water-cured control)
KK     = {"0.50": 0.022488, "0.60": 0.012923}

def parse(fn, col=1):
    t, T = [], []
    for ln in open(fn, encoding="latin-1"):
        tk = ln.split()
        if len(tk) > col:
            try: t.append(float(tk[0])); T.append(float(tk[col]))
            except ValueError: pass
    return np.asarray(t), np.asarray(T)

def teq(t_s, T_C):
    xi = np.exp(-Ea/R*(1.0/(T_C+273.15) - 1.0/Tref_K))
    return np.concatenate([[0.0], np.cumsum(0.5*(xi[1:]+xi[:-1])*np.diff(t_s))]) / 3600.0

def early_T(t_s, T_C, h=72.0):
    m = t_s <= h*3600.0
    return float(T_C[m].mean())

def S_hyp(M, Su, k): return Su*k*M/(1.0+k*M)

print(f"  {'L':>4} | {'peakT50':>7} {'peakT60':>7} {'dT':>5} | {'matr50':>6} {'matr60':>6} | "
      f"{'Su60core':>8} {'Slate60_th':>10}")
rows=[]
for L in SIZES:
    t5,T5 = parse(f"{S050}/sweep_L{L}_core.rpt")
    t6,T6 = parse(f"{S060}/sweep060_L{L}_core.rpt")
    te5, te6 = teq(t5,T5), teq(t6,T6)
    age = t6[-1]/3600.0
    matr5, matr6 = te5[-1]/(t5[-1]/3600.0), te6[-1]/age
    Te6 = early_T(t6,T6)
    Su6 = SU_STD["0.60"]*su_shape(Te6)                 # 0.60 core ultimate strength (thermal crossover)
    cte_late = float(np.interp(min(14*24, age), t6/3600.0, te6))
    Slate60_th = S_hyp(cte_late, Su6, KK["0.60"]) / S_hyp(min(14*24,age), SU_STD["0.60"], KK["0.60"])
    print(f"  {L:>4} | {T5.max():>7.1f} {T6.max():>7.1f} {T6.max()-T5.max():>5.2f} | "
          f"{matr5:>6.2f} {matr6:>6.2f} | {Su6:>8.1f} {Slate60_th:>10.2f}")
    rows.append((L, matr5, matr6, Slate60_th))

print("\n  Thermal field mix-independence:  max |peakT60-peakT50| and |matr60-matr50| across sizes:")
import statistics
dmat = max(abs(r[2]-r[1]) for r in rows)
print(f"     max |d maturity ratio| = {dmat:.3f}  (~0 -> temperature field is mix-independent)")
r400 = [r for r in rows if r[0]==400][0]
print(f"\n  [M4] at 400 mm (specimen): thermal-predicted 0.60 late ratio = {r400[3]:.2f} (a DEFICIT, like 0.50),")
print(f"       but MEASURED 0.60 cube = 1.17-1.27 (SURPLUS).")
print(f"       => moisture (self-desiccation-relief) contribution ~= {1.22 - r400[3]:+.2f}  (non-thermal).")
