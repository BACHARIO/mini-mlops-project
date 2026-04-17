import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Charger le modèle
model = joblib.load("artifacts/model.pkl")

# Structure des données d'entrée
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint santé
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint prédiction
@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    prediction = model.predict(features)[0]
    
    return {"prediction": int(prediction)}