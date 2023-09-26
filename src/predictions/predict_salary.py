import pandas as pd
import logging

LOGGER_FILENAME = "logger.log"

logging.basicConfig(
    filename=LOGGER_FILENAME,
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)


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
    logging.info(result)
    return result
