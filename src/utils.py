# Miscellaneous helper functions.
# src/utils.py
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from pathlib import Path

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def save_model(model, path: Path):
    joblib.dump(model, path)

def load_model(path: Path):
    return joblib.load(path)

import joblib
from pathlib import Path

def save_models(models, folder: Path):
    """Save a list of model dictionaries (as produced by oof_lgb or oof_blend) to folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, model_dict in enumerate(models):
        # We need to ensure all components are serializable. joblib works for most sklearn/lgbm objects.
        joblib.dump(model_dict, folder / f"fold_{i}.pkl")

def load_models(folder: Path):
    """Load saved models from folder."""
    models = []
    for path in sorted(folder.glob("fold_*.pkl")):
        models.append(joblib.load(path))
    return models