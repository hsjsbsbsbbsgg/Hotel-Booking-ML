import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data.config import PROJECT_ROOT, load_config
from src.preprocessing.pipelines import prepare_task_data

CONFIG = load_config()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
SEED = CONFIG["random_state"]
FOLDS = CONFIG["cv_folds"]


def run_baselines(task, models, scoring, cv):
    df = pd.read_csv(PROJECT_ROOT / CONFIG["data_path"])
    X_train, _, y_train, _, preprocessor = prepare_task_data(
        df, task, test_size=CONFIG["test_size"], random_state=SEED
    )
    rows = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", estimator),
        ])
        results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        row = {
            "task": task,
            "model": name,
            "n_train": len(X_train),
            "fit_seconds_per_fold": results["fit_time"].mean(),
        }
        for key in scoring:
            values = results[f"test_{key}"]
            sign = -1 if key.startswith("neg_") else 1
            label = key.replace("neg_", "")
            row[f"cv_{label}"] = sign * values.mean()
            row[f"cv_{label}_std"] = values.std()
        rows.append(row)
        print(rows[-1])
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classification = run_baselines(
        "classification",
        {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=SEED),
            "decision_tree": DecisionTreeClassifier(random_state=SEED),
            "random_forest": RandomForestClassifier(
                n_estimators=100, random_state=SEED, n_jobs=-1
            ),
        },
        {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        StratifiedKFold(FOLDS, shuffle=True, random_state=SEED),
    )
    classification.to_csv(OUTPUT_DIR / "classification_baseline_cv.csv", index=False)

    regression = run_baselines(
        "regression",
        {
            "ridge": Ridge(alpha=1.0, random_state=SEED),
            "decision_tree": DecisionTreeRegressor(random_state=SEED),
            "random_forest": RandomForestRegressor(
                n_estimators=100, random_state=SEED, n_jobs=-1
            ),
        },
        {
            "neg_mean_absolute_error": "neg_mean_absolute_error",
            "neg_root_mean_squared_error": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        KFold(FOLDS, shuffle=True, random_state=SEED),
    )
    regression.to_csv(OUTPUT_DIR / "regression_baseline_cv.csv", index=False)
    print("Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
