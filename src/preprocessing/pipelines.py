from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_sets import (
    ADR_MAX,
    BOOKING_TIME_FEATURES,
    CLUSTERING_FEATURES,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = df.select_dtypes(exclude=["number", "bool"]).columns
    for column in text_columns:
        df[column] = df[column].astype(object).where(df[column].notna(), np.nan)
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    for column in ("Agent", "Company"):
        if column in df.columns:
            df[column] = df[column].replace("NULL", np.nan)
    return df


def get_feature_types(df: pd.DataFrame, features: Sequence[str]):
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    numerical = df[list(features)].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical = [column for column in features if column not in numerical]
    return numerical, categorical


def build_preprocessor(df: pd.DataFrame, features: Sequence[str]) -> ColumnTransformer:
    numerical_features, categorical_features = get_feature_types(df, features)

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    transformers = []
    if numerical_features:
        transformers.append(("numerical", numerical_pipeline, numerical_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_task_data(
    df: pd.DataFrame,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_duplicates: bool = True,
    exclude_features: Sequence[str] | None = None,
):
    if task == "regression":
        target = TARGET_REGRESSION
    elif task == "classification":
        target = TARGET_CLASSIFICATION
    else:
        raise ValueError("task must be either 'regression' or 'classification'")

    features = [
        feature for feature in BOOKING_TIME_FEATURES
        if feature not in set(exclude_features or [])
    ]

    required = set(features) | {target}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    task_df = clean_dataframe(df)[features + [target]]
    task_df = task_df.dropna(subset=[target])
    if task == "regression":
        task_df = task_df[task_df[target] < ADR_MAX]

    if drop_duplicates:
        task_df = task_df.drop_duplicates().reset_index(drop=True)

    X = task_df[features]
    y = task_df[target]

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train, features)
    return X_train, X_test, y_train, y_test, preprocessor


def prepare_clustering_data(df: pd.DataFrame, drop_duplicates: bool = False):
    missing = sorted(set(CLUSTERING_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing clustering columns: {missing}")

    X = clean_dataframe(df)[CLUSTERING_FEATURES].copy()
    if drop_duplicates:
        X = X.drop_duplicates().reset_index(drop=True)
    preprocessor = build_preprocessor(X, CLUSTERING_FEATURES)
    return X, preprocessor
