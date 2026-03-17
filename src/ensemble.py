# src/ensemble.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
from .features import FeatureTransformer
from .config import N_SPLITS, RANDOM_STATE, RIDGE_ALPHA, LGB_PARAMS, BLEND_WEIGHTS
import logging

logger = logging.getLogger(__name__)

def oof_blend(X, y, n_splits=N_SPLITS, lgb_params=None, ridge_alpha=RIDGE_ALPHA,
              blend_weights=BLEND_WEIGHTS, random_state=RANDOM_STATE):
    """
    Perform OOF blending of LightGBM and Ridge.
    Returns oof predictions and a list of fold models for inference.
    """
    if lgb_params is None:
        lgb_params = LGB_PARAMS.copy()
        lgb_params['random_state'] = random_state

    ft = FeatureTransformer()
    oof_preds = np.zeros(len(X))
    models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Fold {fold+1}/{n_splits}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit transformer on training only
        ft.fit(X_train)
        X_train_t = ft.transform(X_train)
        X_val_t = ft.transform(X_val)

        # ---- LightGBM ----
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        # ---- Ridge (with one-hot encoding) ----
        cat_cols = X_train_t.select_dtypes("category").columns.tolist()
        num_cols = X_train_t.select_dtypes(np.number).columns.tolist()

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = ohe.fit_transform(X_train_t[cat_cols])
        X_val_cat = ohe.transform(X_val_t[cat_cols])

        X_train_ridge = np.hstack([X_train_t[num_cols].values, X_train_cat])
        X_val_ridge = np.hstack([X_val_t[num_cols].values, X_val_cat])

        ridge_model = Ridge(alpha=ridge_alpha, random_state=random_state)
        ridge_model.fit(X_train_ridge, y_train)
        ridge_preds_val = ridge_model.predict(X_val_ridge)

        # Blend
        lgb_preds_val = lgb_model.predict(X_val_t)
        blended_val = (blend_weights[0] * lgb_preds_val +
                       blend_weights[1] * ridge_preds_val)
        oof_preds[val_idx] = blended_val

        # Store all needed artifacts for inference
        models.append({
            "lgb": lgb_model,
            "ridge": ridge_model,
            "ohe": ohe,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "ft": ft   # each fold has its own transformer!
        })

        fold_rmse = np.sqrt(mean_squared_error(y_val, blended_val))
        logger.info(f"Fold RMSE: {fold_rmse:.5f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    logger.info(f"Overall OOF RMSE: {overall_rmse:.5f}")
    return oof_preds, models

#  bottom

def oof_lgb(X, y, n_splits=N_SPLITS, lgb_params=None, random_state=RANDOM_STATE):
    """
    Out-of-fold predictions using LightGBM only, with proper per‑fold preprocessing.
    
    Parameters
    ----------
    X : DataFrame
        Raw features (without preprocessing). The function will apply FeatureTransformer
        inside each fold.
    y : Series
        Target. Should be on the original scale (will not be transformed).
    n_splits : int
        Number of folds.
    lgb_params : dict, optional
        Parameters for LightGBM. If None, uses config.LGB_PARAMS.
    random_state : int

    Returns
    -------
    oof_preds : ndarray
        Out-of-fold predictions (on the original scale of y).
    models : list of dict
        Each dict contains {'model': trained LGBM, 'ft': fitted FeatureTransformer}
    """
    if lgb_params is None:
        lgb_params = LGB_PARAMS.copy()
        lgb_params['random_state'] = random_state

    ft = FeatureTransformer()
    oof_preds = np.zeros(len(X))
    models = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit transformer on training only
        ft.fit(X_train)
        X_train_t = ft.transform(X_train)
        X_val_t = ft.transform(X_val)

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        preds = model.predict(X_val_t)
        oof_preds[val_idx] = preds

        models.append({'model': model, 'ft': ft})   # store a fresh transformer per fold

        rmse_fold = np.sqrt(mean_squared_error(y_val, preds))
        logger.info(f"Fold {fold+1} RMSE: {rmse_fold:.5f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    logger.info(f"Overall OOF RMSE: {overall_rmse:.5f}")
    return oof_preds, models