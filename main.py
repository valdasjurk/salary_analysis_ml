import sys, io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from parameters_optimization import (
    find_rfr_best_params_and_score,
    find_linear_regression_best_params,
)
from preprocessor import create_preprocessor
from create_testing_scenarios import create_testing_scenarios, create_predictions_plot
from load_datasets import load_lithuanian_salary_data, load_profession_code_data
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import numpy as np

if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper) and sys.version_info >= (3, 7):
        sys.stdout.reconfigure(encoding="utf-8")

TEST_SIZE = 0.3
XCOLS = [
    "lytis",
    "amzius",
    "profesija",
    "issilavinimas",
    "stazas",
    "darbo_laiko_dalis",
    "svoris",
]
YCOLS = "dbu_metinis"


def create_lr_model() -> Pipeline:
    """Function creates LinearRegression model"""
    preprocesor = create_preprocessor()
    model = Pipeline(
        [
            ("preprocessor", preprocesor),
            ("pol", PolynomialFeatures(degree=3)),
            ("lin", LinearRegression()),
        ]
    )
    return model


def create_rfr_model() -> Pipeline:
    """Function creates RandomForestRegressor model"""
    preprocesor = create_preprocessor()
    model = Pipeline(
        [
            ("preprocessor", preprocesor),
            ("rfr", RandomForestRegressor(n_estimators=10, max_depth=4)),
        ]
    )
    return model


def create_decision_tree_model() -> Pipeline:
    preprocesor = create_preprocessor()
    model = make_pipeline(
        preprocesor, DecisionTreeRegressor(max_depth=4, min_samples_split=2)
    )
    return model


def create_adaboost_model() -> Pipeline:
    preprocesor = create_preprocessor()
    model = make_pipeline(
        preprocesor,
        AdaBoostRegressor(
            estimator=None,
            n_estimators=10,
            learning_rate=1.0,
            # The loss function to use when updating the weights after each boosting iteration.
            loss="linear",
        ),
    )
    return model


def create_lgbm_model() -> Pipeline:
    preprocesor = create_preprocessor()
    model = make_pipeline(
        preprocesor,
        LGBMRegressor(
            num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=5, n_jobs=-1
        ),
    )
    return model


def split_data_to_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X, y = (
        data[XCOLS],
        data[YCOLS],
    )
    return X, y


def add_profession_code_data_to_salary_df(
    org_df: pd.DataFrame, ext_df: pd.DataFrame
) -> pd.DataFrame:
    """Function adds profession description data to original dataframe based on profession number"""
    profession_names = org_df["profesija"].map(
        ext_df.drop_duplicates("Kodas").set_index("Kodas")["Pavadinimas"]
    )
    return org_df.assign(profesijos_apibudinimas=profession_names)


def remove_low_variance_features() -> Pipeline:
    preprocesor = create_preprocessor()
    model = Pipeline(
        [
            ("preprocessor", preprocesor),
            ("rfr", VarianceThreshold(threshold=0.1)),
        ]
    )
    return model


def show_model_feature_importances(model, model_pipeline_name="rfr") -> pd.DataFrame:
    preprocessor_steps = [
        ["num", "imputer"],
        ["cat", "ohe"],
        ["cust", "ohe_cust"],
    ]
    col_names_list = np.array([])
    for i in preprocessor_steps:
        col_names = (
            model.named_steps["preprocessor"]
            .named_transformers_[i[0]][i[1]]
            .get_feature_names_out()
        )
        col_names_list = np.append(col_names_list, col_names)

    feature_importances = rfr_model.named_steps[
        model_pipeline_name
    ].feature_importances_

    df = pd.DataFrame(
        feature_importances,
        index=col_names_list,
    )
    print(df)
    return df


if __name__ == "__main__":
    data = load_lithuanian_salary_data()
    data_ext = load_profession_code_data()

    X, y = split_data_to_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    print("Train test splitted")

    var_thr = remove_low_variance_features()
    var_thr.fit_transform(X_train)

    lr_model = create_lr_model()
    lr_model.fit(X_train, y_train)
    prediction = lr_model.predict(X_test)
    score = lr_model.score(X_test, y_test)
    print("LinearRegression score: ", score)
    joblib.dump(lr_model, "model.joblib")

    rfr_model = create_rfr_model()
    rfr_model.fit(X_train, y_train)
    y_pred = rfr_model.predict(X_test)
    score2 = rfr_model.score(X_test, y_test)
    print("RandomForestRegressor score: ", score2)
    show_model_feature_importances(rfr_model)

    decision_tree_model = create_decision_tree_model()
    decision_tree_model.fit(X_train, y_train)
    prediction3 = decision_tree_model.predict(X_test)
    score3 = decision_tree_model.score(X_test, y_test)
    print("DecisionTreeRegressor score: ", score3)

    adaboost_model = create_adaboost_model()
    decision_tree_model.fit(X_train, y_train)
    prediction_adb = decision_tree_model.predict(X_test)
    score_adb = decision_tree_model.score(X_test, y_test)
    print("AdaBoost score: ", score_adb)

    lgbm_model = create_lgbm_model()
    lgbm_model.fit(X_train, y_train)
    prediction_lgbm = lgbm_model.predict(X_test)
    score_lgbm = lgbm_model.score(X_test, y_test)
    print("LightGBM score: ", score_lgbm)

    # scenarios = create_testing_scenarios()
    # predictions = lr_model.predict(scenarios)
    # predictions_df = scenarios.assign(predictions=predictions)
    # print(predictions_df)

    # img = create_predictions_plot(predictions_df)
    # img.show()

    # find_rfr_best_params_and_score(X_train, y_train, rfr_model)
    # find_linear_regression_best_params(X_train, y_train, lr_model)
