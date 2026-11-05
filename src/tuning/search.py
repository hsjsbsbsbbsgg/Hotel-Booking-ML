import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def build_search(pipeline, space, scoring, cv, n_iter, random_state):
    combinations = int(np.prod([len(values) for values in space.values()]))
    return RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=min(n_iter, combinations),
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        random_state=random_state,
        refit=True,
        verbose=1,
    )
