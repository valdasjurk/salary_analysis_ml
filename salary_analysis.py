import sys

import pandas as pd
from sklearn.compose import ColumnTransformer

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = "https://get.data.gov.lt/datasets/gov/lsd/darbo_uzmokestis/DarboUzmokestis2018/:format/csv"
EXTERNAL_DATA_PATH = "data/raw/profesijos.csv"


def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["_type", "_revision"], axis=1)
    return data.head(5000)


def load_external_data(path=EXTERNAL_DATA_PATH):
    return pd.read_csv(path)


def create_model():
    cat_features = ["lytis"]
    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown="ignore"))

    num_features = ["profesija", "stazas", "darbo_laiko_dalis"]
    num_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )
    model = make_pipeline(
        preprocessor, PolynomialFeatures(degree=3), LinearRegression()
    )
    return model


def predict_salary(
    model, sex: str, profession_name: str, experience: float, workload: float
):
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


def create_testing_scenarios(experience_year=[1, 31]):
    exp_year_start, exp_year_end = experience_year
    param_grid = {
        "lytis": ["F", "M"],
        "stazas": range(exp_year_start, exp_year_end, 3),
        "darbo_laiko_dalis": range(50, 101, 25),
        "profesija": [334],
    }
    return pd.DataFrame(ParameterGrid(param_grid))


def prepare_data(data):
    X, y = (
        data[["lytis", "profesija", "stazas", "darbo_laiko_dalis"]],
        data["dbu_metinis"],
    )
    return train_test_split(X, y, test_size=0.3)


def create_preprocessor():
    cat_features = ["lytis"]
    cat_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])

    num_features = ["profesija", "stazas", "darbo_laiko_dalis"]
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )
    return preprocessor


def find_best_model_parameters(X_train, X_test, y_train, y_test):
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

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=3)
    grid_search.fit(X_train, y_train)

    x = pd.DataFrame(grid_search.cv_results_)
    x = x.set_index("rank_test_score")
    x = x.sort_index()
    return x.loc[:, "param_pol__degree":"param_preprocessor__num__imputer__strategy"]


def add_external_data(org_df, ext_df):
    """Function adds profession description data to original dataframe based on profession number"""
    proffesions_list = []
    for ind in org_df.index:
        proffesions_list.append(
            ext_df.loc[
                ext_df["Kodas"] == org_df["profesija"][ind], "Pavadinimas"
            ].item()
        )
    return org_df.assign(profesijos_apibudinimas=proffesions_list)


if __name__ == "__main__":
    data = load_data()
    data_ext = load_external_data()
    data = add_external_data(data, data_ext)
    X_train, X_test, y_train, y_test = prepare_data(data)
    print("Train test splitted")
    model = create_model()
    print(model)
    model.fit(X_train, y_train)
    print("FITTED")
    prediction = model.predict(X_test)
    print("prediction")
    score = model.score(X_test, y_test)
    print(score)

    scenarios = create_testing_scenarios()
    print(scenarios)
    predictions = model.predict(scenarios)
    r = scenarios.assign(predictions=predictions)
    print(r)

    best_parameters = find_best_model_parameters(X_train, X_test, y_train, y_test)
    print("best parameters:")
    print(best_parameters)
