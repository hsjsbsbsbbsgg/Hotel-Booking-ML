import json
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.config import PROJECT_ROOT, load_config
from src.evaluation.metrics import evaluate_classifier
from src.preprocessing.pipelines import prepare_task_data
from src.tuning.search import build_search
from src.tuning.search_spaces import CLASSIFICATION_SPACES

CONFIG = load_config()
DATA_PATH = PROJECT_ROOT / CONFIG["data_path"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "classification"

RANDOM_STATE = CONFIG["random_state"]
CV_FOLDS = CONFIG["cv_folds"]
N_ITER = CONFIG["n_iter"]
QUICK = CONFIG["quick_mode"]
SCORING = "f1"


def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def make_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(estimator)),
    ])


def run_experiments() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = prepare_task_data(
        df,
        "classification",
        test_size=CONFIG["test_size"],
        random_state=RANDOM_STATE,
    )
    if QUICK:
        X_train = X_train.sample(6000, random_state=RANDOM_STATE)
        y_train = y_train.loc[X_train.index]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, tuned_models = [], {}

    for name, estimator in get_models().items():
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        baseline = make_pipeline(preprocessor, estimator)
        cv_results = cross_validate(
            baseline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=1,
            scoring={"f1": "f1", "roc_auc": "roc_auc"},
        )
        start = time.perf_counter()
        baseline.fit(X_train, y_train)
        baseline_seconds = time.perf_counter() - start
        rows.append({
            "model": name,
            "version": "baseline",
            "cv_f1_mean": cv_results["test_f1"].mean(),
            "cv_f1_std": cv_results["test_f1"].std(),
            "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
            "training_seconds": baseline_seconds,
            **evaluate_classifier(baseline, X_test, y_test),
        })
        print("baseline CV f1:", round(cv_results["test_f1"].mean(), 4))

        search = build_search(
            make_pipeline(preprocessor, estimator),
            CLASSIFICATION_SPACES[name],
            SCORING,
            cv,
            N_ITER,
            RANDOM_STATE,
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        tuning_seconds = time.perf_counter() - start
        best_index = search.best_index_
        rows.append({
            "model": name,
            "version": "tuned",
            "cv_f1_mean": search.best_score_,
            "cv_f1_std": search.cv_results_["std_test_score"][best_index],
            "training_seconds": tuning_seconds,
            "best_params": json.dumps(search.best_params_),
            **evaluate_classifier(search.best_estimator_, X_test, y_test),
        })
        tuned_models[name] = search.best_estimator_
        print("best params:", search.best_params_)
        print("tuned CV f1:", round(search.best_score_, 4))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "classification_model_comparison.csv", index=False)

    tuned = results[results.version == "tuned"].sort_values("cv_f1_mean", ascending=False)
    best_name = tuned.iloc[0]["model"]
    with open(OUTPUT_DIR / "selected_model.json", "w") as file:
        json.dump({
            "selected_model": best_name,
            "criterion": f"highest mean {CV_FOLDS}-fold CV {SCORING} on train set",
            "cv_f1_mean": float(tuned.iloc[0]["cv_f1_mean"]),
            "best_params": tuned.iloc[0]["best_params"],
        }, file, indent=2)

    for name, model in tuned_models.items():
        joblib.dump(model, ARTIFACT_DIR / f"{name}_pipeline.joblib", compress=3)
    joblib.dump(tuned_models[best_name], ARTIFACT_DIR / "best_pipeline.joblib", compress=3)

    print(f"\nSelected (by CV): {best_name}")
    print("Results:", OUTPUT_DIR, "| Artifacts:", ARTIFACT_DIR)
    return results


if __name__ == "__main__":
    output = run_experiments()
    columns = ["model", "version", "cv_f1_mean", "test_f1", "test_roc_auc", "test_pr_auc"]
    print(output[columns].to_string(index=False))
