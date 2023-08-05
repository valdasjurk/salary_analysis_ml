from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUM_FEATURES = ["profesija", "stazas", "darbo_laiko_dalis"]
CAT_FEATURES = ["lytis", "amzius"]


def create_preprocessor() -> ColumnTransformer:
    cat_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore"))])

    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUM_FEATURES),
            ("cat", cat_transformer, CAT_FEATURES),
        ]
    )
    return preprocessor
