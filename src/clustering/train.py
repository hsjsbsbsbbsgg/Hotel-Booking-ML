from __future__ import annotations

import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.data.config import PROJECT_ROOT, load_config
from src.features.feature_sets import (
    CLUSTERING_CATEGORICAL_FEATURES,
    CLUSTERING_FEATURES,
    CLUSTERING_NUMERIC_FEATURES,
)
from src.preprocessing.pipelines import build_preprocessor, clean_dataframe

CONFIG = load_config()
DATA_PATH = PROJECT_ROOT / CONFIG["data_path"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clustering"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "clustering"

RANDOM_STATE = CONFIG["random_state"]
QUICK = CONFIG["quick_mode"]
METRIC_SAMPLE = 2000 if QUICK else 5000
AGG_SAMPLE = 800 if QUICK else 3000
DBSCAN_SAMPLE = 2000 if QUICK else 10000

NUMERIC = CLUSTERING_NUMERIC_FEATURES
CATEGORICAL = CLUSTERING_CATEGORICAL_FEATURES
FEATURES = CLUSTERING_FEATURES


MIN_CLUSTER_FRAC = 0.02
MAX_NOISE_FRAC = 0.30
MIN_CLUSTERS = 3


def scores(X, labels):
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan,
                    n_clusters=n_clusters, noise_frac=float((~mask).mean()),
                    min_cluster_frac=np.nan, valid=False)
    Xm, lm = X[mask], labels[mask]
    sil = silhouette_score(Xm, lm, sample_size=min(METRIC_SAMPLE, len(lm)),
                           random_state=RANDOM_STATE)
    db = davies_bouldin_score(Xm, lm)
    sizes = np.bincount(lm) if lm.min() >= 0 else np.unique(lm, return_counts=True)[1]
    min_frac = float(sizes[sizes > 0].min() / len(lm))
    noise = float((~mask).mean())
    valid = (min_frac >= MIN_CLUSTER_FRAC and noise <= MAX_NOISE_FRAC
             and n_clusters >= MIN_CLUSTERS)
    return dict(silhouette=sil, davies_bouldin=db, n_clusters=n_clusters,
                noise_frac=noise, min_cluster_frac=min_frac, valid=valid)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    df = clean_dataframe(raw)[FEATURES]
    pre = build_preprocessor(df, FEATURES)
    X_all = pre.fit_transform(df)
    X_all = X_all.toarray() if hasattr(X_all, "toarray") else np.asarray(X_all)
    joblib.dump(pre, ARTIFACT_DIR / "preprocessor.joblib", compress=3)
    print("Prepared:", X_all.shape)

    rng = np.random.RandomState(RANDOM_STATE)
    idx_agg = rng.choice(len(X_all), min(AGG_SAMPLE, len(X_all)), replace=False)
    idx_db = rng.choice(len(X_all), min(DBSCAN_SAMPLE, len(X_all)), replace=False)
    X_agg, X_db = X_all[idx_agg], X_all[idx_db]

    rows, labels_store = [], {}

    def record(algo, version, params, X, labels):
        row = dict(algorithm=algo, version=version, params=json.dumps(params),
                   n_samples=len(X), **scores(X, labels))
        rows.append(row)
        labels_store[(algo, version, json.dumps(params))] = np.asarray(labels)
        return row


    km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit(X_all)
    record("KMeans", "baseline", {"n_clusters": 3}, X_all, km.labels_)
    for k in range(2, 9):
        m = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_all)
        record("KMeans", "grid", {"n_clusters": k}, X_all, m.labels_)


    record("Agglomerative", "baseline", {"n_clusters": 3, "linkage": "ward"},
           X_agg, AgglomerativeClustering(n_clusters=3).fit_predict(X_agg))
    for link in ["ward", "complete", "average"]:
        for k in range(2, 9):
            lab = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X_agg)
            record("Agglomerative", "grid", {"n_clusters": k, "linkage": link}, X_agg, lab)


    record("DBSCAN", "baseline", {"eps": 0.5, "min_samples": 5},
           X_db, DBSCAN(eps=0.5, min_samples=5).fit_predict(X_db))
    for eps in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        for ms in [5, 10, 20]:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_db)
            record("DBSCAN", "grid", {"eps": eps, "min_samples": ms}, X_db, lab)

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "clustering_comparison.csv", index=False)


    best = {}
    for algo, g in res[res.version == "grid"].groupby("algorithm"):
        valid = g[g.valid]
        pool = valid if len(valid) else g.dropna(subset=["silhouette"])
        if len(pool):
            best[algo] = pool.sort_values("silhouette", ascending=False).iloc[0].to_dict()
    pd.DataFrame(best).T.to_csv(OUTPUT_DIR / "best_per_algorithm.csv")
    print(pd.DataFrame(best).T[["version", "params", "n_clusters", "silhouette",
                                "davies_bouldin", "min_cluster_frac"]].to_string())


    k_best = int(json.loads(best["KMeans"]["params"])["n_clusters"])
    final = KMeans(n_clusters=k_best, n_init=20, random_state=RANDOM_STATE).fit(X_all)
    joblib.dump(final, ARTIFACT_DIR / "kmeans_final.joblib", compress=3)
    prof = clean_dataframe(raw).copy()
    prof["cluster"] = final.labels_
    prof["IsCanceled"] = raw["IsCanceled"].values
    num_prof = prof.groupby("cluster")[NUMERIC + ["IsCanceled"]].mean().round(2)
    num_prof["size"] = prof.groupby("cluster").size()
    num_prof.to_csv(OUTPUT_DIR / "cluster_numeric_profile.csv")
    pd.concat({c: prof.groupby("cluster")[c].agg(lambda s: s.value_counts().idxmax())
               for c in CATEGORICAL}, axis=1).to_csv(OUTPUT_DIR / "cluster_categorical_mode.csv")
    print("\nFinal KMeans k =", k_best)
    print(num_prof.to_string())


    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    sel = rng.choice(len(X_all), min(8000, len(X_all)), replace=False)
    Z = pca.fit_transform(X_all[sel])
    plt.figure(figsize=(7, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=final.labels_[sel], s=4, cmap="tab10")
    plt.title(f"KMeans (k={k_best}) - PCA 2D "
              f"(explained {pca.explained_variance_ratio_.sum():.1%})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_pca_2d.png", dpi=150); plt.close()

    sub = X_agg[: min(300, len(X_agg))]
    plt.figure(figsize=(10, 5))
    dendrogram(linkage(sub, method="ward"), no_labels=True, truncate_mode="lastp", p=30)
    plt.title("Ward dendrogram (sample)"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "agglomerative_dendrogram.png", dpi=150); plt.close()

    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(dict(rows=len(raw), features=FEATURES, encoded_shape=list(X_all.shape),
                       agglomerative_sample=len(X_agg), dbscan_sample=len(X_db),
                       metric_sample=METRIC_SAMPLE,
                       validity_rule=dict(min_cluster_frac=MIN_CLUSTER_FRAC,
                                          max_noise_frac=MAX_NOISE_FRAC,
                                          min_clusters=MIN_CLUSTERS)), f, indent=2)


if __name__ == "__main__":
    main()
