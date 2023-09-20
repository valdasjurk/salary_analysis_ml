from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_testing_scenarios(
    experience_year=[1, 31], profession=[251, 334], age=["40-49"], education=["G4"]
) -> pd.DataFrame:
    exp_year_start, exp_year_end = experience_year
    param_grid = {
        "lytis": ["F", "M"],
        "stazas": range(exp_year_start, exp_year_end, 3),
        "darbo_laiko_dalis": range(50, 101, 25),
        "profesija": profession,
        "amzius": age,
        "issilavinimas": education,
    }
    return pd.DataFrame(ParameterGrid(param_grid))


def plot_predictions(data: pd.DataFrame, save=True):
    sns.scatterplot(
        x="stazas",
        y="predictions",
        data=data,
        hue="lytis",
        size="darbo_laiko_dalis",
        style="profesija",
        palette=["red", "blue"],
    )
    plt.xlabel("Work experience, years")
    plt.ylabel("Predicted yearly salary, eur")
    if save is True:
        plt.savefig("reports/figures/testing_predictions.png")
    else:
        plt.show()
    return plt
