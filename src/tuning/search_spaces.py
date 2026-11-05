CLASSIFICATION_SPACES = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
    },
    "decision_tree": {
        "model__max_depth": [5, 8, 12, 16, 20, 30, None],
        "model__min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "model__min_samples_split": [2, 10, 30],
        "model__ccp_alpha": [0.0, 1e-4, 5e-4, 1e-3],
        "model__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20, 30],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.3],
        "model__class_weight": [None, "balanced"],
    },
}

REGRESSION_SPACES = {
    "ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "decision_tree": {
        "model__max_depth": [5, 8, 12, 16, 20, 30, None],
        "model__min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "model__min_samples_split": [2, 10, 30],
        "model__ccp_alpha": [0.0, 1e-3, 1e-2, 1e-1],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 15, 20, 30],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
    },
}
