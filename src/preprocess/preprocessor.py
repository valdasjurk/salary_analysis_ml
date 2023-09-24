from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from preprocess.custom_profession_name_transformer import ProfessionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

NUM_FEATURES = ["profesija", "stazas", "darbo_laiko_dalis"]
CAT_FEATURES = ["lytis", "amzius", "issilavinimas"]
CUSTOM_FEATURES = ["profesija"]
TFIDF_FEATURES = "profesijos_apibudinimas"


def create_preprocessor() -> Pipeline:
    cat_transformer = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    cust_transformer = Pipeline(
        steps=[
            ("prof_name", ProfessionTransformer()),
        ]
    )
    tfidf_transformer = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words=["ir"])),
        ]
    )
    num_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUM_FEATURES),
            ("cat", cat_transformer, CAT_FEATURES),
            ("text", tfidf_transformer, TFIDF_FEATURES),
        ]
    )
    preprocessor = Pipeline(
        steps=[("cust", cust_transformer), ("col_tran", col_transformer)]
    )
    return preprocessor
