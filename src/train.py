# src/train.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from .cv import get_folds
from .features import log_transform_target, inverse_log_transform
from .config import LOG_TARGET, TARGET, N_SPLITS, RANDOM_STATE
import logging

logger = logging.getLogger(__name__)

def train_model_oof(model, X, y, fit_params=None, log_target=False):
    """
    Perform cross-validation and return out-of-fold predictions.
    
    Parameters
    ----------
    model : callable or estimator instance
        If callable, it should return a fresh estimator. If estimator, it will be cloned per fold.
    X : DataFrame
        Features (already preprocessed).
    y : Series
        Target. If log_target=True, y is assumed to be raw target and will be log-transformed.
        If log_target=False, y is used as is (should be already transformed if desired).
    fit_params : dict, optional
        Additional parameters passed to the model's fit method (e.g., eval_set for early stopping).
    log_target : bool, default=False
        Whether to apply log1p transformation to the target.

    Returns
    -------
    oof_preds : ndarray
        Out-of-fold predictions (on the same scale as the input y, i.e., if log_target=True,
        predictions are on log scale; you need to inverse transform separately).
    scores : list
        RMSE scores for each fold (calculated on the scale of y after possible log transform).
    """
    if log_target:
        y_trans = log_transform_target(y)
    else:
        y_trans = y

    folds = get_folds(n_splits=N_SPLITS, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(folds.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_trans.iloc[train_idx], y_trans.iloc[val_idx]

        # Create a fresh model for this fold
        if callable(model):
            m = model()
        else:
            # Assume it's an unfitted estimator; clone to be safe
            from sklearn.base import clone
            m = clone(model)

        # Prepare fit arguments
        fit_kwargs = fit_params.copy() if fit_params else {}
        # If eval_set is in fit_params, we need to pass the validation data correctly
        if 'eval_set' in fit_kwargs:
            # The user provided eval_set as a list of (X, y) tuples? Usually it's a list of (X_val, y_val)
            # For simplicity, we override eval_set with the current validation fold.
            fit_kwargs['eval_set'] = [(X_val, y_val)]

        m.fit(X_train, y_train, **fit_kwargs)
        preds = m.predict(X_val)
        oof_preds[val_idx] = preds

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        logger.info(f"Fold {fold+1} RMSE: {rmse:.5f}")

    logger.info(f"Mean RMSE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return oof_preds, scores