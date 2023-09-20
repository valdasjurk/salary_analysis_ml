from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from preprocess.custom_profession_name_transformer import ProfessionTransformer

NUM_FEATURES = ["profesija", "stazas", "darbo_laiko_dalis"]
CAT_FEATURES = ["lytis", "amzius", "issilavinimas"]
CUSTOM_FEATURES = ["profesija"]


def create_preprocessor() -> ColumnTransformer:
    cat_transformer = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    cust_transformer = Pipeline(
        steps=[
            ("prof_name", ProfessionTransformer()),
            ("ohe_cust", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUM_FEATURES),
            ("cust", cust_transformer, CUSTOM_FEATURES),
            ("cat", cat_transformer, CAT_FEATURES),
        ]
    )
    return preprocessor
