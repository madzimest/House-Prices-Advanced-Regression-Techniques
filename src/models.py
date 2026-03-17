# src/models.py
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from .config import LGB_PARAMS, RIDGE_ALPHA, RANDOM_STATE

def get_ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE):
    return Ridge(alpha=alpha, random_state=random_state)

def get_rf(n_estimators=500, random_state=RANDOM_STATE):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

def get_lgb(params=None):
    if params is None:
        params = LGB_PARAMS.copy()
    return lgb.LGBMRegressor(**params)