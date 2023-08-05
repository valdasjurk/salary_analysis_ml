from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_testing_scenarios(experience_year=[1, 31]) -> pd.DataFrame:
    exp_year_start, exp_year_end = experience_year
    param_grid = {
        "lytis": ["F", "M"],
        "stazas": range(exp_year_start, exp_year_end, 3),
        "darbo_laiko_dalis": range(50, 101, 25),
        "profesija": [334],
        "amzius": ["40-49"],
    }
    return pd.DataFrame(ParameterGrid(param_grid))


def create_predictions_plot(data: pd.DataFrame):
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
