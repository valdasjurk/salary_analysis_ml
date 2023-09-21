import shap
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = "rfr"  # RandomForestRegressor


def parse_x_column_names(model):
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
    return col_names_list


def plot_shap_importances(model, X_train, y_train, show_or_save="show"):
    # explain the model's predictions using SHAP
    column_names = parse_x_column_names(model)
    model.fit(X_train, y_train)
    model_masker = model[:-1].transform(X_train).toarray()

    explainer = shap.Explainer(
        model.named_steps[MODEL_NAME],
        masker=model_masker,
        feature_names=column_names,
    )
    shap_values = explainer(model_masker)
    fig = shap.plots.beeswarm(shap_values, max_display=15, show=False)
    if show_or_save == "show":
        plt.show()
    else:
        plt.savefig("reports/figures/shap.png")
    return fig
