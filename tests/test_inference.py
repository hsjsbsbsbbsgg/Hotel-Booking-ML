import joblib
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.inference.predict import predict_booking
from src.preprocessing.pipelines import prepare_task_data


def save_pipeline(bookings, task, estimator, directory):
    X_train, _, y_train, _, preprocessor = prepare_task_data(bookings, task)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, directory / "best_pipeline.joblib")


def test_classification_prediction(bookings, example_booking, tmp_path):
    save_pipeline(bookings, "classification", LogisticRegression(max_iter=500), tmp_path)
    result = predict_booking("classification", example_booking, artifacts_dir=tmp_path)
    assert result["is_canceled"] in (0, 1)
    assert 0.0 <= result["cancel_probability"] <= 1.0


def test_regression_prediction(bookings, example_booking, tmp_path):
    save_pipeline(bookings, "regression", Ridge(), tmp_path)
    result = predict_booking("regression", example_booking, artifacts_dir=tmp_path)
    assert isinstance(result["adr"], float)


def test_missing_field_raises(bookings, example_booking, tmp_path):
    save_pipeline(bookings, "regression", Ridge(), tmp_path)
    del example_booking["LeadTime"]
    with pytest.raises(ValueError):
        predict_booking("regression", example_booking, artifacts_dir=tmp_path)


def test_missing_artifact_raises(example_booking, tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_booking("classification", example_booking, artifacts_dir=tmp_path)
