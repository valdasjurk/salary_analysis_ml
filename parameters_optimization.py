from sklearn.model_selection import RandomizedSearchCV


RANDOM_FOREST_REGRESSOR_PARAMS = {
    "preprocessor__num__imputer__strategy": ["most_frequent", "mean"],
    "rfr__n_estimators": [4, 5, 6, 7, 8, 9, 10, 20],
    "rfr__max_depth": [2, 3, 4, 5],
    "rfr__min_samples_leaf": [1, 2],
    "rfr__bootstrap": [True, False],
}


def find_best_params_and_score_with_randomizedsearchcv(X_train, y_train, model):
    randomized_search = RandomizedSearchCV(
        estimator=model, param_distributions=RANDOM_FOREST_REGRESSOR_PARAMS
    )
    randomized_search.fit(X_train, y_train)
    print(
        "Selected model best parameters (based on Randomized search): ",
        randomized_search.best_params_,
        randomized_search.best_score_,
    )
