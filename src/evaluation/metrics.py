import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_f1": f1_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
        "test_pr_auc": average_precision_score(y_test, probabilities),
        "confusion_matrix": json.dumps(confusion_matrix(y_test, predictions).tolist()),
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return {
        "test_mae": mean_absolute_error(y_test, predictions),
        "test_mse": mse,
        "test_rmse": float(np.sqrt(mse)),
        "test_r2": r2_score(y_test, predictions),
    }
