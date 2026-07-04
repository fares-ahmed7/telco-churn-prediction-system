import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==========================================================
# Preprocessor
# ==========================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build the preprocessing pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer.
    """

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


# ==========================================================
# Full Pipeline
# ==========================================================

def build_pipeline(preprocessor, model) -> Pipeline:
    """
    Build a complete machine learning pipeline.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing transformer.

    model : sklearn estimator
        Machine learning model.

    Returns
    -------
    Pipeline
    """

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ==========================================================
# Models
# ==========================================================

def get_models():
    """
    Return all candidate models.
    """

    return {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    }