# src/features.py
import logging
import numpy as np
import pandas as pd
from scipy.stats import skew
from typing import Optional, List
from .config import SKEW_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_transform_target(y: pd.Series) -> pd.Series:
    return np.log1p(y)

def inverse_log_transform(y: pd.Series) -> pd.Series:
    return np.expm1(y)

class FeatureTransformer:
    """
    Handles all preprocessing: missing values, feature engineering,
    skew correction, and categorical conversion.
    Fit on training data, then transform train/test.
    """
    def __init__(self, convert_cats: bool = True):
        self.convert_cats = convert_cats
        self._cat_cols: Optional[List[str]] = None
        self._num_cols: Optional[List[str]] = None
        self._skewed_feats: Optional[List[str]] = None
        self._lot_frontage_medians: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "FeatureTransformer":
        """Learn necessary parameters from training data."""
        df = df.copy()

        # Store column types for later
        self._cat_cols = df.select_dtypes(include="object").columns.tolist()
        self._num_cols = df.select_dtypes(include=np.number).columns.tolist()

        # For LotFrontage, compute median per neighborhood (global fallback)
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            self._lot_frontage_medians = df.groupby("Neighborhood")["LotFrontage"].median()
        else:
            self._lot_frontage_medians = None

        # Identify skewed numeric features (only positive, to avoid log of negative)
        pos_num_cols = [c for c in self._num_cols if (df[c] > 0).all()]
        if pos_num_cols:
            skewed = df[pos_num_cols].apply(lambda x: skew(x.dropna()))
            self._skewed_feats = skewed[abs(skewed) > SKEW_THRESHOLD].index.tolist()
        else:
            self._skewed_feats = []

        logger.info(f"Skewed features to transform: {self._skewed_feats}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to any dataframe."""
        df = df.copy()
        df = self._handle_missing(df)
        df = self._add_features(df)
        df = self._correct_skew(df)
        if self.convert_cats:
            df = self._convert_categoricals(df)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Columns that should be filled with "None"
        none_cols = [
            "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"
        ]
        for col in none_cols:
            if col in df.columns:
                df[col] = df[col].fillna("None")

        # LotFrontage: use neighborhood medians from fit
        if "LotFrontage" in df.columns and self._lot_frontage_medians is not None:
            df["LotFrontage"] = df.apply(
                lambda row: self._lot_frontage_medians.get(row["Neighborhood"], np.nan)
                if pd.isna(row["LotFrontage"]) else row["LotFrontage"],
                axis=1
            )
            # If still NaN (e.g., new neighborhood in test), fill with global median
            df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

        # Fill remaining numeric missing with median
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Safely add features only if required columns exist
        if all(c in df.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
            df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        if all(c in df.columns for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
            df["TotalBathrooms"] = (df["FullBath"] + 0.5*df["HalfBath"] +
                                     df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"])
        if all(c in df.columns for c in ["YrSold", "YearBuilt"]):
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        if all(c in df.columns for c in ["YrSold", "YearRemodAdd"]):
            df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
        if all(c in df.columns for c in ["OverallQual", "TotalSF"]):
            df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
        return df

    def _correct_skew(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._skewed_feats:
            for col in self._skewed_feats:
                # log1p works only for non-negative values; ensure we don't break
                if col in df.columns and (df[col] >= 0).all():
                    df[col] = np.log1p(df[col])
                else:
                    logger.warning(f"Column {col} not suitable for log transform, skipping.")
        return df

    def _convert_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("category")
        return df