from sklearn.base import BaseEstimator, TransformerMixin
from load_datasets import load_profession_code_data


class ProfessionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        profession_codes_df = load_profession_code_data()
        transformed_X = X.assign(
            profesijos_apibudinimas=(
                X["profesija"].map(
                    profession_codes_df.drop_duplicates("Kodas").set_index("Kodas")[
                        "Pavadinimas"
                    ]
                )
            ),
            profesijos_apibudinimas_2=(
                X["profesija"].astype(str).apply(lambda x: x[:2])
            )
            .astype(int)
            .map(
                profession_codes_df.drop_duplicates("Kodas").set_index("Kodas")[
                    "Pavadinimas"
                ]
            ),
            profesijos_apibudinimas_3=(X["profesija"].astype(str).apply(lambda x: x[0]))
            .astype(int)
            .map(
                profession_codes_df.drop_duplicates("Kodas").set_index("Kodas")[
                    "Pavadinimas"
                ]
            ),
        )
        return transformed_X


def test_stuff():
    a = ProfessionTransformer()
    import pandas as pd
    from pandas.testing import assert_frame_equal

    df = pd.DataFrame({"profesija": [324]})
    assert_frame_equal(
        a.transform(df),
        pd.DataFrame(
            [
                [
                    324,
                    "Veterinarijos technikai ir felčeriai",
                    "Jaunesnieji sveikatos specialistai",
                    "Technikai ir jaunesnieji specialistai",
                ]
            ],
            columns=[
                "profesija",
                "profesijos_apibudinimas",
                "profesijos_apibudinimas_2",
                "profesijos_apibudinimas_3",
            ],
        ),
    )
