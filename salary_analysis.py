import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from parameters_optimization import find_best_params_with_randomizedsearchcv
from preprocessor import create_preprocessor

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
    model = make_pipeline(preprocesor, PolynomialFeatures(degree=3), LinearRegression())
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


def create_testing_scenarios(experience_year=[1, 31]) -> pd.DataFrame:
    exp_year_start, exp_year_end = experience_year
    param_grid = {
        "lytis": ["F", "M"],
        "stazas": range(exp_year_start, exp_year_end, 3),
        "darbo_laiko_dalis": range(50, 101, 25),
        "profesija": [334],
    }
    return pd.DataFrame(ParameterGrid(param_grid))


def plot_predictions(data: pd.DataFrame):
    sns.scatterplot(
        x="stazas",
        y="predictions",
        data=data,
        hue="lytis",
        size="darbo_laiko_dalis",
        palette=["red", "blue"],
    )
    plt.xlabel("Work experience, years")
    plt.ylabel("Predicted yearly salary, eur")
    return plt


def split_data_to_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X, y = (
        data[XCOLS],
        data[YCOLS],
    )
    return X, y


def create_pipeline_and_parameters_for_gridsearchcv(
    X_train: pd.DataFrame, y_train: pd.DataFrame
):
    preprocesor = create_preprocessor()
    pipeline = Pipeline(
        [
            ("preprocessor", preprocesor),
            ("pol", PolynomialFeatures(degree=3)),
            ("lin", LinearRegression()),
        ]
    )
    parameters = {
        "preprocessor__num__imputer__strategy": ["most_frequent", "mean"],
        "pol__degree": [1, 2, 3, 4, 5],
    }
    return pipeline, parameters


def transform_cv_results(cv_results: dict) -> pd.DataFrame:
    df = pd.DataFrame(cv_results)
    df = df.set_index("rank_test_score")
    df = df.sort_index()
    return df.loc[:, df.columns[4] : df.columns[-7]]


def add_profession_code_data_to_salary_df(
    org_df: pd.DataFrame, ext_df: pd.DataFrame
) -> pd.DataFrame:
    """Function adds profession description data to original dataframe based on profession number"""
    profession_names = org_df["profesija"].map(
        ext_df.drop_duplicates("Kodas").set_index("Kodas")["Pavadinimas"]
    )
    return org_df.assign(profesijos_apibudinimas=profession_names)


if __name__ == "__main__":
    data = load_lithuanian_salary_data()
    data_ext = load_profession_code_data()
    data = add_profession_code_data_to_salary_df(data, data_ext)

    X, y = split_data_to_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    print("Train test splitted")

    model = create_lr_model()
    model.fit(X_train, y_train)
    print("FITTED")
    joblib.dump(model, "model.joblib")
    prediction = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("LinearRegression score: ", score)

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

    find_best_params_with_randomizedsearchcv(X_train, y_train, model2)

    scenarios = create_testing_scenarios()
    print(scenarios)
    predictions = model.predict(scenarios)
    r = scenarios.assign(predictions=predictions)
    print(r)

    img = plot_predictions(r)
    img.show()

    (
        gridsearchcv_pipeline,
        gridsearchcv_parameters,
    ) = create_pipeline_and_parameters_for_gridsearchcv(X_train, y_train)
    grid_search = GridSearchCV(
        gridsearchcv_pipeline, gridsearchcv_parameters, n_jobs=-1, cv=3
    )
    grid_search.fit(X_train, y_train)
    cv_results = grid_search.cv_results_
    best_parameters = transform_cv_results(cv_results)
    print("best parameters:")
    print(best_parameters)
