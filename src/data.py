# src/data.py
import pandas as pd
from .config import TRAIN_FILE, TEST_FILE

def load_train():
    """Load training data."""
    return pd.read_csv(TRAIN_FILE)

def load_test():
    """Load test data."""
    return pd.read_csv(TEST_FILE)

def load_raw_data():
    """Return train and test as a tuple."""
    return load_train(), load_test()