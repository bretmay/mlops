# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

class Instance(BaseModel):
    features: list

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(inst: Instance):
    X = np.array(inst.features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    return {"prediction": pred}
