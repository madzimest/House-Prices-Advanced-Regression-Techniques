# Cross-validation strategies and utilities.
# src/cv.py
from sklearn.model_selection import KFold, StratifiedKFold
from .config import RANDOM_STATE, N_SPLITS

def get_folds(y=None, n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True):
    """
    Returns a KFold instance. If y is provided and is categorical,
    could switch to StratifiedKFold (but for regression we stay with simple KFold).
    """
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)