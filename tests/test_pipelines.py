import numpy as np
import pandas as pd
import pytest

from src.features.feature_sets import BOOKING_TIME_FEATURES, EXCLUDED_FEATURES
from src.preprocessing.pipelines import clean_dataframe, prepare_task_data


def row_set(X, y):
    frame = X.assign(target=y.values).astype(object).fillna("NA").astype(str)
    return set(map(tuple, frame.values))


def test_clean_dataframe_strips_text_and_marks_null(bookings):
    cleaned = clean_dataframe(bookings)
    assert set(cleaned["Meal"].unique()) <= {"BB", "HB", "SC"}
    assert "NULL" not in set(cleaned["Agent"].dropna().unique())
    assert cleaned["Company"].isna().any()


def test_excluded_features_are_never_used(bookings):
    assert not set(EXCLUDED_FEATURES) & set(BOOKING_TIME_FEATURES)
    for task in ("classification", "regression"):
        X_train, X_test, _, _, _ = prepare_task_data(bookings, task)
        assert not set(EXCLUDED_FEATURES) & set(X_train.columns)
        assert list(X_train.columns) == list(X_test.columns) == BOOKING_TIME_FEATURES


def test_duplicates_are_removed_before_split(bookings):
    X_train, X_test, y_train, y_test, _ = prepare_task_data(bookings, "classification")
    unique_rows = len(
        clean_dataframe(bookings)[BOOKING_TIME_FEATURES + ["IsCanceled"]].drop_duplicates()
    )
    assert len(X_train) + len(X_test) == unique_rows

    assert not row_set(X_train, y_train) & row_set(X_test, y_test)


def test_classification_split_is_stratified(bookings):
    _, _, y_train, y_test, _ = prepare_task_data(bookings, "classification")
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)


def test_regression_drops_adr_outlier(bookings):
    outlier = bookings.head(1).copy()
    outlier["ADR"] = 5400.0
    outlier["LeadTime"] = 999
    data = pd.concat([bookings, outlier], ignore_index=True)
    _, _, y_train, y_test, _ = prepare_task_data(data, "regression")
    assert max(y_train.max(), y_test.max()) < 1000


def test_preprocessor_handles_unseen_category(bookings):
    X_train, X_test, _, _, preprocessor = prepare_task_data(bookings, "classification")
    transformed_train = preprocessor.fit_transform(X_train)
    unseen = X_test.copy()
    unseen["Country"] = "ZZZ"
    transformed_unseen = preprocessor.transform(unseen)
    assert transformed_unseen.shape[1] == transformed_train.shape[1]
    dense = transformed_unseen.toarray() if hasattr(transformed_unseen, "toarray") else transformed_unseen
    assert not np.isnan(dense).any()


def test_unknown_task_raises(bookings):
    with pytest.raises(ValueError):
        prepare_task_data(bookings, "other")
