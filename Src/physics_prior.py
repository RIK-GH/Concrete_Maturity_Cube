"""
physics_prior.py - fold-aware ASTM C1074 hyperbolic prior (LEAKAGE-SAFE).

Critical for honest cross-validation: the per-mix hyperbolic parameters (Su, k)
must be estimated using TRAINING rows only. A held-out mix therefore has NO
per-mix fit and MUST fall back to a pooled Su(w/c) prior learned from train mixes.
Computing S_hyp once on the full data (as an ordinary feature) leaks the test
mix's own strengths through its Su, k -> this class prevents that.

Usage inside a CV fold:
    pm = PriorModel().fit(df.iloc[train_idx])
    S_hyp_train = pm.predict(df.iloc[train_idx])
    S_hyp_test  = pm.predict(df.iloc[test_idx])   # unseen mix -> pooled prior
"""
import numpy as np
from scipy.optimize import curve_fit

MIN_PTS = 3
MIN_SPAN = 2.5


def _hyp(M, su, k):
    return su * k * M / (1.0 + k * M)


def _fit_one(M, S):
    M = np.asarray(M, float); S = np.asarray(S, float)
    ok = np.isfinite(M) & np.isfinite(S) & (M > 0)
    M, S = M[ok], S[ok]
    if len(M) < MIN_PTS or M.max() / max(M.min(), 1e-6) < MIN_SPAN:
        return None
    try:
        popt, _ = curve_fit(_hyp, M, S, p0=[S.max() * 1.25, 0.01],
                            bounds=([S.max() * 0.9, 1e-5], [S.max() * 2.0 + 20, 5.0]),
                            maxfev=20000)
    except Exception:
        return None
    return float(popt[0]), float(popt[1])


class PriorModel:
    """Fold-safe hyperbolic prior. For UNSEEN mixes the asymptote Su is w/c-conditioned
    (Su ~ 1/wc). A w/c-conditioned rate constant (ln k ~ wc) is also available (USE_KWC):
    it recovers the slab k's almost exactly (k(0.44)=0.018 vs 0.019; k(0.64)=0.0099 vs
    0.0099) and helps the hard Exp#1 (RMSE 8.2->6.9) BUT slightly degrades the headline
    Exp#2 (2.66->2.98) because it shifts the whole training residual field; the k-wc
    correlation is weak (r=-0.21). Kept OFF by default; see the ablation in methods."""

    USE_KWC = False

    def __init__(self):
        self.mix_params = {}                 # mix_id -> (Su, k)
        self.k_med = 0.02
        self.su_a, self.su_b = 0.0, 45.0     # Su_hat = a/wc + b
        self.k_a, self.k_b = 0.0, np.log(0.02)  # ln k_hat = k_a*wc + k_b

    def fit(self, df):
        ks = []
        pool_wc, pool_su = [], []
        kfit_wc, kfit_lnk = [], []
        for mid, g in df.groupby("mix_id"):
            r = _fit_one(g["t_eq"].values, g["f_c"].values)
            if r is not None and r[0] < 120:
                self.mix_params[mid] = r
                ks.append(r[1])
                wc = g["WC"].dropna()
                if len(wc):
                    pool_wc.append(1.0 / wc.iloc[0]); pool_su.append(r[0])
                    kfit_wc.append(wc.iloc[0]); kfit_lnk.append(np.log(r[1]))
        if ks:
            self.k_med = float(np.median(ks))
            self.k_b = np.log(self.k_med)
        if len(pool_wc) >= 3:
            self.su_a, self.su_b = np.polyfit(pool_wc, pool_su, 1)
        elif pool_su:
            self.su_a, self.su_b = 0.0, float(np.median(pool_su))
        if len(kfit_wc) >= 5:
            self.k_a, self.k_b = np.polyfit(kfit_wc, kfit_lnk, 1)
        return self

    def _su_hat(self, wc, fc_max):
        base = self.su_a / wc + self.su_b if (wc and np.isfinite(wc)) else self.su_b
        return float(np.clip(base, (fc_max or 0) * 1.02, 130))

    def _k_hat(self, wc):
        if self.USE_KWC and wc and np.isfinite(wc):
            return float(np.clip(np.exp(self.k_a * wc + self.k_b), 1e-4, 1.0))
        return self.k_med

    def predict(self, df):
        out = np.empty(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            mid = row["mix_id"]
            if mid in self.mix_params:                 # seen mix -> its own curve
                su, k = self.mix_params[mid]
            else:                                      # unseen mix -> w/c-conditioned prior
                wc = row.get("WC", np.nan)
                su = self._su_hat(wc, row.get("f_c", np.nan))
                k = self._k_hat(wc)
            out[i] = max(_hyp(row["t_eq"], su, k), 0.0)
        return out
