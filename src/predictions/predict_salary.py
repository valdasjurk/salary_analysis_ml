import pandas as pd
import numpy as np


def predict_salary(
    model,
    sex: str,
    age: str,
    profession_code: int,
    experience: int,
    workload: float,
    education: str,
) -> dict:
    data = {
        "lytis": sex,
        "amzius": age,
        "profesija": profession_code,
        "issilavinimas": education,
        "stazas": experience,
        "darbo_laiko_dalis": workload,
    }
    z = pd.DataFrame([data])
    result = {"result": {"Yearly salary prediction, eur": round(model.predict(z)[0])}}
    print(result)
    return result
