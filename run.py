import os
import sys
import argparse
from typing import List

sys.path.append(os.path.join(os.getcwd(), "src/"))
from sklearn.model_selection import train_test_split

from src.load_datasets import load_lithuanian_salary_data
from src.main import (
    compare_lr_scikit_to_torch,
    create_lr_model,
    create_rfr_model,
    fit_model_and_show_score,
    remove_low_variance_features,
    split_data_to_xy,
)
from src.predictions.create_testing_scenarios import (
    create_testing_scenarios,
    plot_predictions,
)
from src.visualization.shap_importances import plot_shap_importances
from src.predictions.predict_salary import predict_salary


def create_lr_model_and_show_score():
    X_train, X_test, y_train, y_test = load_and_split_dataset()

    var_threshold = remove_low_variance_features()
    var_threshold.fit_transform(X_train)
    model = create_lr_model()
    fit_model_and_show_score(model, X_train, y_train, X_test, y_test)
    return model


def create_rfr_model_and_show_score():
    X_train, X_test, y_train, y_test = load_and_split_dataset()

    var_threshold = remove_low_variance_features()
    var_threshold.fit_transform(X_train)
    model = create_rfr_model()
    fit_model_and_show_score(model, X_train, y_train, X_test, y_test)
    return model


def compare_lr_scikit_to_torch_by_mse():
    X_train, X_test, y_train, y_test = load_and_split_dataset()
    compare_lr_scikit_to_torch(X_train, y_train, X_test, y_test)


def create_testing_scenarios_and_predict(
    experience_year, profession, age, education, show
):
    X_train, _, y_train, _ = load_and_split_dataset()

    scenarios = create_testing_scenarios(experience_year, profession, age, education)
    lr_model = create_lr_model()
    lr_model.fit(X_train, y_train)
    print(scenarios)
    predictions = lr_model.predict(scenarios)
    predictions_df = scenarios.assign(predictions=predictions)
    plot_predictions(predictions_df, bool(show))


def load_and_split_dataset():
    data = load_lithuanian_salary_data()
    X, y = split_data_to_xy(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def shap_feature_importances(show):
    X_train, _, y_train, _ = load_and_split_dataset()

    model = create_rfr_model_and_show_score()
    plot_shap_importances(model, X_train, y_train, bool(show))


def predict_yearly_salary(sex, age, profession_name, experience, workload, education):
    X_train, _, y_train, _ = load_and_split_dataset()
    model = create_lr_model()
    model.fit(X_train, y_train)
    predict_salary(model, sex, age, profession_name, experience, workload, education)


if __name__ == "__main__":
    # Get our arguments from the user
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--create_lr_model_and_show_score",
        help="Crete LinearRegression model",
        action="store_true",
    )
    parser.add_argument(
        "--create_rfr_model_and_show_score",
        help="Crete RandomForestRegressor model",
        action="store_true",
    )
    parser.add_argument(
        "--compare_lr_scikit_to_torch_by_mse",
        action="store_true",
    )
    parser.add_argument(
        "--create_testing_scenarios_and_predict",
        action="store_true",
    )
    parser.add_argument(
        "--experience_year", type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument(
        "--profession", type=lambda s: [int(item) for item in s.split(",")]
    )
    parser.add_argument(
        "--age_group", type=lambda s: [str(item) for item in s.split(",")]
    )
    parser.add_argument(
        "--education", type=lambda s: [str(item) for item in s.split(",")]
    )
    parser.add_argument("--show")
    parser.add_argument(
        "--shap_feature_importances",
        action="store_true",
    )
    parser.add_argument(
        "--predict_yearly_salary",
        action="store_true",
    )
    parser.add_argument("--sex", type=str)
    parser.add_argument("--profession_code", type=int)
    parser.add_argument("--exp", type=int)
    parser.add_argument("--workload", type=int)
    parser.add_argument("--age", type=str)
    parser.add_argument("--educ", type=str)

    args = parser.parse_args()

    if args.create_lr_model_and_show_score:
        create_lr_model_and_show_score()
    if args.create_lr_model_and_show_score:
        create_rfr_model_and_show_score()
    if args.compare_lr_scikit_to_torch_by_mse:
        compare_lr_scikit_to_torch_by_mse()
    if args.create_testing_scenarios_and_predict:
        create_testing_scenarios_and_predict(
            args.experience_year,
            args.profession,
            args.age_group,
            args.education,
            args.show,
        )
    if args.shap_feature_importances:
        shap_feature_importances(args.show)
    if args.predict_yearly_salary:
        predict_yearly_salary(
            args.sex, args.age, args.profession_code, args.exp, args.workload, args.educ
        )
