import pandas as pd
from sklearn.compose import ColumnTransformer

# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

DATA_PATH = "https://get.data.gov.lt/datasets/gov/lsd/darbo_uzmokestis/DarboUzmokestis2018/:format/csv"


def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["_type", "_revision"], axis=1)
    return data.head(5000)


def create_model(X, y):
    cat_features = ["lytis"]
    cat_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
    )

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

    tts = train_test_split(X, y, test_size=0.3)
    X_train, X_test, y_train, y_test = tts

    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


def predict_salary(
    model, sex: str, profession_name: str, experience: float, workload: float
):
    """Sex = [M, F] , profession_name = gydytojas, experience in years, workload in percentage 0-100%"""
    profession_dict = {"administratorius": 334}
    z = pd.DataFrame(
        [
            {
                "lytis": sex,
                "profesija": profession_dict[profession_name],
                "stazas": experience,
                "darbo_laiko_dalis": workload,
            }
        ]
    )
    result = {"result": {"Yearly salary prediction, eur": round(model.predict(z)[0])}}
    print(result)
    return result


def predict_salary_various_scenarios(
    model, profession_name: str, experience_year=[1, 10]
):
    """Profession_name = gydytojas; experience_year = [1, 10] (egzample).
    Function returns dictionary with yearly salary predictions of given profession and year interval for male and female
    """
    profession_dict = {"administratorius": 334}
    result = {"result": {}}
    for sex in ["M", "F"]:
        for i in range(experience_year[0], experience_year[1]):
            z = pd.DataFrame(
                [
                    {
                        "lytis": sex,
                        "profesija": profession_dict[profession_name],
                        "stazas": i,
                        "darbo_laiko_dalis": 100,
                    }
                ]
            )
            new_dict = {
                f"Yearly salary prediction, eur [sex: {sex}, experience: {i}]": round(
                    model.predict(z)[0]
                )
            }
            result["result"].update(new_dict)
    return result


if __name__ == "__main__":
    data = load_data()
    X, y = (
        data[["lytis", "profesija", "stazas", "darbo_laiko_dalis"]],
        data["dbu_metinis"],
    )

    model, score = create_model(X, y)
    print(score)
    # predict_salary(modelis, "F", "gydytojas", 1, 100)
    # scenarios = predict_salary_various_scenarios(model, "administratorius", experience_year=[1, 10])
    # print(scenarios)
