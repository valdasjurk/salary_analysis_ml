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
from create_testing_scenarios import create_testing_scenarios, plot_predictions
from load_datasets import load_lithuanian_salary_data, load_profession_code_data
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from torch_linear_regression import create_torch_lr_model_and_show_loss
from shap_importances import plot_shap_importances, parse_x_column_names

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
        steps=[
            ("preprocessor", preprocesor),
            ("pol", PolynomialFeatures(degree=3)),
            ("lin", LinearRegression()),
        ]
    )
    print("LinearRegression model created")
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
    print("RandomForestRegressor model created")
    return model


def create_decision_tree_model() -> Pipeline:
    preprocesor = create_preprocessor()
    model = make_pipeline(
        preprocesor, DecisionTreeRegressor(max_depth=4, min_samples_split=2)
    )
    print("DecisionTreeRegressor model created")
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
    print("AdaBoostRegressor model created")
    return model


def create_lgbm_model() -> Pipeline:
    preprocesor = create_preprocessor()
    model = make_pipeline(
        preprocesor,
        LGBMRegressor(
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.1,
            n_estimators=5,
            n_jobs=-1,
            force_col_wise=True,
        ),
    )
    print("LGBMRegressor model created")
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
    x_column_names = parse_x_column_names(model)
    feature_importances = model.named_steps[model_pipeline_name].feature_importances_
    df = pd.DataFrame(
        list(zip(x_column_names, feature_importances)),
        columns=["Feature", "Importance"],
    )
    df_sorted = df.sort_values(by=["Importance"], ascending=False).head(20)
    return df_sorted


def fit_model_and_show_score(model):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("Model score: ", score)
    return score


if __name__ == "__main__":
    data = load_lithuanian_salary_data()
    data_ext = load_profession_code_data()

    X, y = split_data_to_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    print("Train test splitted")

    var_thr = remove_low_variance_features()
    var_thr.fit_transform(X_train)

    lr_model = create_lr_model()
    fit_model_and_show_score(lr_model)
    joblib.dump(lr_model, "model.joblib")

    rfr_model = create_rfr_model()
    fit_model_and_show_score(rfr_model)

    # plot_shap_importances(rfr_model, X_train, y_train)
    show_model_feature_importances(rfr_model)

    # decision_tree_model = create_decision_tree_model()
    # fit_model_and_show_score(decision_tree_model)

    # lgbm_model = create_lgbm_model()
    # fit_model_and_show_score(lgbm_model)

    # scenarios = create_testing_scenarios()
    # predictions = lr_model.predict(scenarios)
    # predictions_df = scenarios.assign(predictions=predictions)
    # img = plot_predictions(predictions_df)

    # find_rfr_best_params_and_score(X_train, y_train, rfr_model)
    # find_linear_regression_best_params(X_train, y_train, lr_model)

    create_torch_lr_model_and_show_loss(X_train, y_train, X_test, y_test)
