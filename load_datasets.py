import pandas as pd

DATA_PATH = "https://get.data.gov.lt/datasets/gov/lsd/darbo_uzmokestis/DarboUzmokestis2018/:format/csv"
EXTERNAL_DATA_PATH = "data/raw/profesijos.csv"
SALARY_DATASET_MAX_ROWS = 5000


def load_lithuanian_salary_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["_type", "_revision"], axis=1)
    if SALARY_DATASET_MAX_ROWS:
        return data.head(SALARY_DATASET_MAX_ROWS)
    return data


def load_profession_code_data(path=EXTERNAL_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
