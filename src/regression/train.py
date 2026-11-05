from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data.config import PROJECT_ROOT, load_config
from src.evaluation.metrics import evaluate_regressor
from src.preprocessing.pipelines import prepare_task_data
from src.tuning.search import build_search
from src.tuning.search_spaces import REGRESSION_SPACES

CONFIG = load_config()
DATA_PATH = PROJECT_ROOT / CONFIG["data_path"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regression"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "regression"

RANDOM_STATE = CONFIG["random_state"]
CV_FOLDS = CONFIG["cv_folds"]
N_ITER = CONFIG["n_iter"]
QUICK = CONFIG["quick_mode"]
SCORING = "neg_root_mean_squared_error"


def get_models() -> dict:
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def make_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(estimator)),
    ])


def run_experiments(models: list[str] | None = None) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test, preprocessor = prepare_task_data(
        df,
        "regression",
        test_size=CONFIG["test_size"],
        random_state=RANDOM_STATE,
    )
    if QUICK:
        X_train = X_train.sample(6000, random_state=RANDOM_STATE)
        y_train = y_train.loc[X_train.index]
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, tuned_models = [], {}

    for name, estimator in get_models().items():
        if models and name not in models:
            continue
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

        baseline = make_pipeline(preprocessor, estimator)
        cv_results = cross_validate(
            baseline,
            X_train,
            y_train,
            cv=cv,
            n_jobs=1,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
        start = time.perf_counter()
        baseline.fit(X_train, y_train)
        baseline_seconds = time.perf_counter() - start
        rows.append({
            "model": name,
            "version": "baseline",
            "cv_rmse_mean": -cv_results["test_rmse"].mean(),
            "cv_rmse_std": cv_results["test_rmse"].std(),
            "cv_r2_mean": cv_results["test_r2"].mean(),
            "training_seconds": baseline_seconds,
            **evaluate_regressor(baseline, X_test, y_test),
        })
        print("baseline CV rmse:", round(-cv_results["test_rmse"].mean(), 3))

        search = build_search(
            make_pipeline(preprocessor, estimator),
            REGRESSION_SPACES[name],
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
            "cv_rmse_mean": -search.best_score_,
            "cv_rmse_std": search.cv_results_["std_test_score"][best_index],
            "training_seconds": tuning_seconds,
            "best_params": json.dumps(search.best_params_),
            **evaluate_regressor(search.best_estimator_, X_test, y_test),
        })
        tuned_models[name] = search.best_estimator_
        print("best params:", search.best_params_)
        print("tuned CV rmse:", round(-search.best_score_, 3))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "regression_model_comparison.csv", index=False)

    tuned = results[results.version == "tuned"].sort_values("cv_rmse_mean")
    best_name = tuned.iloc[0]["model"]
    with open(OUTPUT_DIR / "selected_model.json", "w") as file:
        json.dump({
            "selected_model": best_name,
            "criterion": f"lowest mean {CV_FOLDS}-fold CV RMSE on train set",
            "cv_rmse_mean": float(tuned.iloc[0]["cv_rmse_mean"]),
            "best_params": tuned.iloc[0]["best_params"],
        }, file, indent=2)

    for name, model in tuned_models.items():
        joblib.dump(model, ARTIFACT_DIR / f"{name}_pipeline.joblib", compress=3)
    joblib.dump(tuned_models[best_name], ARTIFACT_DIR / "best_pipeline.joblib", compress=3)

    print(f"\nSelected (by CV): {best_name}")
    return results


if __name__ == "__main__":
    output = run_experiments()
    columns = ["model", "version", "cv_rmse_mean", "test_mae", "test_rmse", "test_r2"]
    print(output[columns].to_string(index=False))
