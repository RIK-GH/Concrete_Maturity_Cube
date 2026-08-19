"""
maturity.py - Unified maturity / equivalent-age / hydration-degree calculators.

All three data sources are put on ONE convention here:
  * Nurse-Saul maturity      M_NS = sum( (T - T0) * dt )           [deg C * h]
  * Arrhenius equivalent age t_eq = sum( exp(-Ea/R (1/T - 1/Tref)) * dt ) [h]
  * Degree-day               DD   = sum( (T) * dt ) / 24           [deg C * day]  (T0=0)

For ISOTHERMAL literature points (constant curing temperature over the whole age)
the sums collapse to closed forms, used for the 55-point and 503-point pools.
For measured CORE-temperature histories (slab FBG, cube TC sensors) the trapezoidal
integrators `integrate_*` consume a (time[h], temp[C]) series - these plug in when
the raw 2-min logs arrive.
"""
import numpy as np
from config import R_GAS, T0_NS, TREF_K, EA_DEFAULT, ALPHA_INF, BETA_HYD


# --------------------------------------------------------------------------
# Closed-form ISOTHERMAL calculators (constant T over age t hours)
# --------------------------------------------------------------------------
def nurse_saul_iso(temp_c, age_h, t0=T0_NS):
    """Nurse-Saul maturity [deg C * h] for constant temperature."""
    return (np.asarray(temp_c, float) - t0) * np.asarray(age_h, float)


def eq_age_iso(temp_c, age_h, ea=EA_DEFAULT, tref_k=TREF_K, r=R_GAS):
    """Arrhenius equivalent age [h] at reference temperature, constant T."""
    tk = np.asarray(temp_c, float) + 273.15
    factor = np.exp(-ea / r * (1.0 / tk - 1.0 / tref_k))
    return factor * np.asarray(age_h, float)


def degree_day_iso(temp_c, age_h, t0=0.0):
    """Degree-day maturity [deg C * day] with datum t0 (paper's reported column: t0=0)."""
    return (np.asarray(temp_c, float) - t0) * np.asarray(age_h, float) / 24.0


# --------------------------------------------------------------------------
# TRAPEZOIDAL integrators for measured temperature histories
#   time_h : 1-D array of timestamps in hours (monotone)
#   temp_c : 1-D array of temperature in deg C at those timestamps
# Returns the cumulative quantity at each timestamp (same length as input).
# --------------------------------------------------------------------------
def _cum_trapz(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    dcum = np.zeros_like(y)
    dcum[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return dcum


def integrate_nurse_saul(time_h, temp_c, t0=T0_NS):
    return _cum_trapz(np.asarray(temp_c, float) - t0, time_h)


def integrate_eq_age(time_h, temp_c, ea=EA_DEFAULT, tref_k=TREF_K, r=R_GAS):
    tk = np.asarray(temp_c, float) + 273.15
    rate = np.exp(-ea / r * (1.0 / tk - 1.0 / tref_k))
    return _cum_trapz(rate, time_h)


def integrate_degree_day(time_h, temp_c, t0=0.0):
    return _cum_trapz(np.asarray(temp_c, float) - t0, time_h) / 24.0


def eq_age_weighted_iso(temp_c, age_h, gamma, ea=EA_DEFAULT, tref_k=TREF_K, r=R_GAS):
    """Crossover-weighted equivalent age (isothermal). Multiplies the Arrhenius rate by a
    quality factor phi(T)=exp(-gamma*max(0, T-Tref)) that discounts maturity accrued ABOVE
    the reference temperature - representing the coarser, weaker C-S-H formed under hot
    early curing (the physical cause of the temperature-crossover effect). gamma=0 recovers
    the standard Arrhenius equivalent age."""
    tk = np.asarray(temp_c, float) + 273.15
    tref_c = tref_k - 273.15
    rate = np.exp(-ea / r * (1.0 / tk - 1.0 / tref_k))
    phi = np.exp(-gamma * np.clip(np.asarray(temp_c, float) - tref_c, 0, None))
    return rate * phi * np.asarray(age_h, float)


def integrate_eq_age_weighted(time_h, temp_c, gamma, ea=EA_DEFAULT, tref_k=TREF_K, r=R_GAS):
    """Crossover-weighted equivalent age for a measured temperature history (trapezoidal)."""
    temp_c = np.asarray(temp_c, float)
    tk = temp_c + 273.15
    tref_c = tref_k - 273.15
    rate = np.exp(-ea / r * (1.0 / tk - 1.0 / tref_k))
    phi = np.exp(-gamma * np.clip(temp_c - tref_c, 0, None))
    return _cum_trapz(rate * phi, time_h)


# --------------------------------------------------------------------------
# ASTM C1074 (Annex A1) apparent-activation-energy calibration.
#
# Procedure implemented:
#   1. For each isothermal curing temperature T_i, fit the age-based hyperbolic
#      strength gain  S(t) = Su * k * (t - t0) / (1 + k * (t - t0))
#      to obtain the rate constant k_i (t0 = age at start of strength gain; here
#      taken 0 for literature points as final-set age is not reported).
#   2. Regress ln(k_i) against 1/T_i(K).  slope = -Ea/R  ->  Ea = -slope * R.
#
# Requires strength-age data at >= 2 temperatures for the SAME mix. Mixes A/B/C
# in the 55-point pool have 5/20/40 C -> 3 Arrhenius points each.
# --------------------------------------------------------------------------
from scipy.optimize import curve_fit


def _hyperbolic_age(t, su, k):
    return su * k * t / (1.0 + k * t)


def fit_rate_constant(age_h, strength, p0_su=None):
    """Fit S = Su*k*t/(1+k*t); return (Su, k, r2). Age in hours."""
    age_h = np.asarray(age_h, float)
    strength = np.asarray(strength, float)
    m = np.isfinite(age_h) & np.isfinite(strength) & (age_h > 0)
    age_h, strength = age_h[m], strength[m]
    if len(age_h) < 3:
        return np.nan, np.nan, np.nan
    su0 = p0_su if p0_su else max(strength) * 1.3
    try:
        popt, _ = curve_fit(_hyperbolic_age, age_h, strength,
                            p0=[su0, 0.01], maxfev=20000,
                            bounds=([strength.max() * 0.8, 1e-5], [200, 5.0]))
    except Exception:
        return np.nan, np.nan, np.nan
    su, k = popt
    pred = _hyperbolic_age(age_h, su, k)
    ss_res = np.sum((strength - pred) ** 2)
    ss_tot = np.sum((strength - strength.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return su, k, r2


def calibrate_Ea(mix_rows, r=R_GAS):
    """
    mix_rows: dict {curing_temp_C: (age_h_array, strength_array)} for ONE mix.
    Returns dict with Ea [J/mol], intercept, R2 of the Arrhenius plot, and per-T k.
    """
    temps, lnk, kdict = [], [], {}
    for tC, (age, strg) in mix_rows.items():
        su, k, r2 = fit_rate_constant(age, strg)
        kdict[tC] = (k, r2)
        if np.isfinite(k) and k > 0:
            temps.append(1.0 / (tC + 273.15))
            lnk.append(np.log(k))
    if len(temps) < 2:
        return {"Ea": np.nan, "r2_arrhenius": np.nan, "k_by_T": kdict, "n_temp": len(temps)}
    temps = np.asarray(temps)
    lnk = np.asarray(lnk)
    slope, intercept = np.polyfit(temps, lnk, 1)     # ln k = -Ea/R * (1/T) + c
    ea = -slope * r
    pred = slope * temps + intercept
    ss_res = np.sum((lnk - pred) ** 2)
    ss_tot = np.sum((lnk - lnk.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"Ea": ea, "intercept": intercept, "r2_arrhenius": r2,
            "k_by_T": kdict, "n_temp": len(temps)}


# --------------------------------------------------------------------------
# Hydration degree (physics-derived feature for the hybrid ML, T3)
# --------------------------------------------------------------------------
def hydration_degree_bazant(M, tau=0.30, alpha_inf=ALPHA_INF, beta=BETA_HYD):
    """Bazant/paper Eq.13:  alpha(M) = alpha_inf * (M/(tau+M))**beta. M in days."""
    M = np.asarray(M, float)
    return alpha_inf * (M / (tau + M)) ** beta


def hydration_degree_fh(te, tau=0.30, alpha_inf=ALPHA_INF, beta=BETA_HYD):
    """Freiesleben-Hansen:  alpha(te) = alpha_inf * exp(-(tau/te)**beta). te in days."""
    te = np.asarray(te, float)
    with np.errstate(divide="ignore"):
        return alpha_inf * np.exp(-((tau / np.where(te > 0, te, np.nan)) ** beta))
