import io
import sys

import logging
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from preprocess.preprocessor import create_preprocessor
from torch_linear_regression import create_torch_lr_model_and_show_loss
from visualization.shap_importances import (
    parse_x_column_names,
)

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
LOGGER_FILENAME = "logger.log"

logging.basicConfig(
    filename=LOGGER_FILENAME,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)


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
    logging.info("LinearRegression model created")
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
    logging.info("RandomForestRegressor model created")
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
    logging.info(f"Model feature importances: {df_sorted}")
    return df_sorted


def fit_model_and_show_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Model accuracy: {score}")
    logging.info(f"Model MSEs: {mse}")
    return mse


def compare_lr_scikit_to_torch(X_train, y_train, X_test, y_test):
    lr_model = create_lr_model()
    mse_scikit = int(
        fit_model_and_show_score(lr_model, X_train, y_train, X_test, y_test)
    )
    X_train = X_train.drop(["profesijos_apibudinimas"], axis=1)
    X_test = X_test.drop(["profesijos_apibudinimas"], axis=1)
    mse_torch, _ = create_torch_lr_model_and_show_loss(X_train, y_train, X_test, y_test)
    mse_torch = int(mse_torch)

    if mse_scikit < mse_torch:
        logging.info(
            f"Scikit learn LinearRegression model is more accurate with a MSE value {mse_scikit} compared to pyTorch nn.Linear regression MSE value {mse_torch}"
        )
    else:
        logging.info(
            f"pyTorch nn.Linear regression model is more accurate with a MSE value {mse_torch} compared to Scikit learn LinearRegression MSE value {mse_scikit}"
        )
