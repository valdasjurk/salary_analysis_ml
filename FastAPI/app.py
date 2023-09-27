import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src/"))
from preprocess.preprocessor import create_preprocessor


class Salary(BaseModel):
    lytis: str
    profesija: int
    issilavinimas: str
    stazas: int
    darbo_laiko_dalis: int
    amzius: str


app = FastAPI()


@app.on_event("startup")
async def load_model():
    global model
    model = joblib.load("model.joblib")


@app.post("/predict")
def predict(salary: Salary):
    data = pd.DataFrame([dict(salary)])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}
