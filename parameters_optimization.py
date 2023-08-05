import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

RANDOM_FOREST_REGRESSOR_PARAMS = {
    "preprocessor__num__imputer__strategy": ["most_frequent", "mean"],
    "rfr__n_estimators": [4, 5, 6, 7, 8, 9, 10, 20],
    "rfr__max_depth": [2, 3, 4, 5],
    "rfr__min_samples_leaf": [1, 2],
    "rfr__bootstrap": [True, False],
}

LINEAR_REGRESSION_PARAMS = {
    "preprocessor__num__imputer__strategy": ["most_frequent", "mean"],
    "pol__degree": [1, 2, 3, 4, 5],
}


def find_rfr_best_params_and_score(X_train, y_train, model):
    randomized_search = RandomizedSearchCV(
        estimator=model, param_distributions=RANDOM_FOREST_REGRESSOR_PARAMS
    )
    randomized_search.fit(X_train, y_train)
    print(
        "Selected model best parameters (based on Randomized search): ",
        randomized_search.best_params_,
        randomized_search.best_score_,
    )


def find_linear_regression_best_params(
    X_train: pd.DataFrame, y_train: pd.DataFrame, model
):
    grid_search = GridSearchCV(model, LINEAR_REGRESSION_PARAMS, n_jobs=-1, cv=3)
    grid_search.fit(X_train, y_train)
    cv_results = grid_search.cv_results_
    best_parameters = transform_cv_results(cv_results)
    print("best parameters:")
    print(best_parameters)


def transform_cv_results(cv_results: dict) -> pd.DataFrame:
    df = pd.DataFrame(cv_results)
    df = df.set_index("rank_test_score")
    df = df.sort_index()
    return df.loc[:, df.columns[4] : df.columns[-7]]
