import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data.config import PROJECT_ROOT, load_config
from src.preprocessing.pipelines import prepare_task_data

CONFIG = load_config()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification"
SEED = CONFIG["random_state"]
N_ESTIMATORS = 30 if CONFIG["quick_mode"] else 100

SETTINGS = {
    "with_duplicates": dict(drop_duplicates=False),
    "no_duplicates (final)": dict(drop_duplicates=True),
    "no_duplicates, no_DepositType": dict(
        drop_duplicates=True, exclude_features=["DepositType"]
    ),
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / CONFIG["data_path"])
    rows = []
    for name, options in SETTINGS.items():
        X_train, X_test, y_train, y_test, preprocessor = prepare_task_data(
            df,
            "classification",
            test_size=CONFIG["test_size"],
            random_state=SEED,
            **options,
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)),
        ]).fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        rows.append({
            "setting": name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })
        print(rows[-1])
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "leakage_ablation.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
