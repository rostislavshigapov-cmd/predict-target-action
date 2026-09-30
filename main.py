import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from data_loading import prepare_data
from config import *

bundle = joblib.load("model/make_target_action.pkl")
model = bundle["model"]

class TargetAction(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str

class Prediction(BaseModel):
    Prediction: int
    Probability: str

app = FastAPI()

@app.get("/status")
def status():
    return "I'm OK"

@app.get("/version")
def version():
    return bundle['metadata']

@app.post("/predict", response_model=Prediction)
def predict(form: TargetAction):
    t = pd.DataFrame([form.dict()])
    t = prepare_data(t)
    t = t.reindex(columns=feature_cols, fill_value=0)
    y = model.predict(t)[0]
    proba_zero = model.predict_proba(t)[0][0]
    proba_one = model.predict_proba(t)[0][1]
    return {
        "Prediction": y,
        "Probability": f"Probability 0: {(proba_zero * 100):.2f}%, Probability 1: {(proba_one * 100):.2f}%"
    }
