import shap
import numpy as np
import matplotlib.pyplot as plt


def parse_x_column_names(model):
    column_names = (
        model.named_steps["preprocessor"]
        .named_steps["col_tran"]
        .get_feature_names_out()
    )
    return column_names


def plot_shap_importances(model, X_train, y_train, show=False):
    # explain the model's predictions using SHAP
    model.fit(X_train, y_train)
    model_name = model.steps[-1][0]
    column_names = parse_x_column_names(model)
    model_masker = model[:-1].transform(X_train).toarray()

    explainer = shap.Explainer(
        model.named_steps[model_name],
        masker=model_masker,
        feature_names=column_names,
    )
    shap_values = explainer(model_masker)
    fig = shap.plots.beeswarm(shap_values, max_display=15, show=False)
    if show is True:
        plt.show()
    else:
        plt.savefig("reports/figures/shap.png")
    return fig
