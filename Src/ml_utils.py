"""ml_utils.py - shared model factory, feature sets, metrics, CV helpers."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE

# Feature groups ------------------------------------------------------------
# paper-style maturity features (Table 6 derived): DD, sqrt(DD), log time, eq.age, NS
FEAT_MATURITY = ["DD", "sqrt_DD", "log_time", "t_eq", "M_NS"]
# mix-design descriptors
FEAT_MIX = ["WC", "C", "RH"]
# physics-derived (T3): hydration degree + mechanistic prior
FEAT_PHYS = ["alpha_bazant", "alpha_fh", "S_hyp"]

FEATURES_PAPER = FEAT_MATURITY + FEAT_MIX          # reproduces the paper's inputs
FEATURES_HYBRID = FEAT_MATURITY + FEAT_MIX + FEAT_PHYS


def make_models():
    """The paper's six algorithms (same hyperparameters where stated)."""
    return {
        "Ridge_a1.0": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Ridge_a0.1": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=0.1))]),
        "Lasso_a0.5": Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.5, max_iter=50000))]),
        "ElasticNet": Pipeline([("sc", StandardScaler()), ("m", ElasticNet(alpha=0.5, max_iter=50000))]),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "GradientBoost": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2); ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(yt), 1e-6, None)) * 100)
    return dict(RMSE=rmse, MAE=mae, R2=r2, MAPE=mape, n=int(m.sum()))


def cv_predict(model, X, y, cv):
    """Out-of-fold predictions honoring an arbitrary CV splitter (indices)."""
    import copy
    oof = np.full(len(y), np.nan)
    for tr, te in cv:
        mdl = copy.deepcopy(model)
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict(X[te])
    return oof


def loo_splits(n):
    for i in range(n):
        yield np.array([j for j in range(n) if j != i]), np.array([i])


def group_splits(groups):
    """Leave-one-group-out on an array of group labels."""
    groups = np.asarray(groups)
    for g in pd.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        yield tr, te
