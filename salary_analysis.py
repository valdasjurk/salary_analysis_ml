import sys

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

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "https://get.data.gov.lt/datasets/gov/lsd/darbo_uzmokestis/DarboUzmokestis2018/:format/csv"
EXTERNAL_DATA_PATH = "data/raw/profesijos.csv"
SALARY_DATASET_MAX_ROWS = 5000
TEST_SIZE = 0.3
XCOLS = ["lytis", "profesija", "stazas", "darbo_laiko_dalis", "amzius"]
YCOLS = "dbu_metinis"
NUM_FEATURES = ["profesija", "stazas", "darbo_laiko_dalis"]
CAT_FEATURES = ["lytis", "amzius"]


def load_lithuanian_salary_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["_type", "_revision"], axis=1)
    if SALARY_DATASET_MAX_ROWS:
        return data.head(SALARY_DATASET_MAX_ROWS)
    return data


def load_profession_code_data(path=EXTERNAL_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


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
    # model = make_pipeline(preprocesor, PolynomialFeatures(degree=3), LinearRegression())
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


def predict_salary(
    model, sex: str, profession_name: str, experience: float, workload: float
) -> dict:
    """Sex = [M, F] , profession_name = gydytojas, experience in years, workload in percentage 0-100%"""
    profession_dict = {"administratorius": 334}
    data = {
        "lytis": sex,
        "profesija": profession_dict[profession_name],
        "stazas": experience,
        "darbo_laiko_dalis": workload,
    }
    z = pd.DataFrame([data])
    result = {"result": {"Yearly salary prediction, eur": round(model.predict(z)[0])}}
    print(result)
    return result


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


# def create_pipeline_and_parameters_for_gridsearchcv(
#     X_train: pd.DataFrame, y_train: pd.DataFrame
# ):
#     preprocesor = create_preprocessor()
#     pipeline = Pipeline(
#         [
#             ("preprocessor", preprocesor),
#             ("pol", PolynomialFeatures(degree=3)),
#             ("lin", LinearRegression()),
#         ]
#     )
#     parameters = {
#         "preprocessor__num__imputer__strategy": ["most_frequent", "mean"],
#         "pol__degree": [1, 2, 3, 4, 5],
#     }
#     return pipeline, parameters


if __name__ == "__main__":
    data = load_lithuanian_salary_data()
    data_ext = load_profession_code_data()
    data = add_profession_code_data_to_salary_df(data, data_ext)

    X, y = split_data_to_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    print("Train test splitted")

    model = create_lr_model()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("LinearRegression score: ", score)
    joblib.dump(model, "model.joblib")

    model2 = create_rfr_model()
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    score2 = model2.score(X_test, y_test)
    print("RandomForestRegressor score: ", score2)

    model3 = create_decision_tree_model()
    model3.fit(X_train, y_train)
    prediction3 = model3.predict(X_test)
    score3 = model3.score(X_test, y_test)
    print("DecisionTreeRegressor score: ", score3)

    model_adb = create_adaboost_model()
    model_adb.fit(X_train, y_train)
    prediction_adb = model_adb.predict(X_test)
    score_adb = model_adb.score(X_test, y_test)
    print("AdaBoost score: ", score_adb)

    model_lgbm = create_lgbm_model()
    model_lgbm.fit(X_train, y_train)
    prediction_lgbm = model_lgbm.predict(X_test)
    score_lgbm = model_lgbm.score(X_test, y_test)
    print("LightGBM score: ", score_lgbm)

    scenarios = create_testing_scenarios()
    predictions = model.predict(scenarios)
    predictions_df = scenarios.assign(predictions=predictions)
    print(predictions_df)

    img = create_predictions_plot(predictions_df)
    img.show()

    find_rfr_best_params_and_score(X_train, y_train, model2)
    find_linear_regression_best_params(X_train, y_train, model)
