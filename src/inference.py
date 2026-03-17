# src/inference.py
import numpy as np
from .features import inverse_log_transform
from .config import BLEND_WEIGHTS
import logging

logger = logging.getLogger(__name__)

def predict_test(models, test_df):
    """
    models: list of dicts as produced by oof_blend (each contains 'lgb', 'ridge', 'ohe',
            'cat_cols', 'num_cols', 'ft')
    """
    final_preds = np.zeros(len(test_df))

    for i, model_dict in enumerate(models):
        ft = model_dict["ft"]
        X_test_t = ft.transform(test_df.copy())

        # LGB predictions
        lgb_preds = model_dict["lgb"].predict(X_test_t)

        # Ridge predictions
        cat_cols = model_dict["cat_cols"]
        num_cols = model_dict["num_cols"]
        ohe = model_dict["ohe"]

        test_cat = ohe.transform(X_test_t[cat_cols])
        test_ridge = np.hstack([X_test_t[num_cols].values, test_cat])
        ridge_preds = model_dict["ridge"].predict(test_ridge)

        # Blend using weights from config
        blended = BLEND_WEIGHTS[0] * lgb_preds + BLEND_WEIGHTS[1] * ridge_preds
        final_preds += blended

    final_preds /= len(models)
    return inverse_log_transform(final_preds)