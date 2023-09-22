from sklearn.base import BaseEstimator, TransformerMixin
from load_datasets import load_profession_code_data


class ProfessionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        profession_codes_df = load_profession_code_data()
        X["profesijos_apibudinimas"] = X["profesija"].map(
            profession_codes_df.drop_duplicates("Kodas").set_index("Kodas")[
                "Pavadinimas"
            ]
        )
        return X
