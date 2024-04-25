from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# Define a Pydantic model to validate the request body
class HouseFeatures(BaseModel):
    OverallQual: float
    GarageCars: float
    GrLivArea: float
    MasVnrArea: float
    FullBath: float
    TotalBsmtSF: float
    YearBuilt: float
    YearRemodAdd: float
    Fireplaces: float
    OpenPorchSF: float
    LotFrontage: float
    WoodDeckSF: float
    BsmtFinSF1: float


app = FastAPI()

# Load your machine learning model from the pickle file
with open("catboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


@app.get('/api/healthchecker')
def root():
    return {'message': 'Welcome'}


@app.post('/api/predict')
async def predict(features: HouseFeatures):
    """
    Endpoint to make predictions using the loaded ML model.
    Expects a JSON payload with the features required for prediction.
    """
    # Extract features from the request data and format them for prediction
    features_list = [features.OverallQual, features.GarageCars, features.GrLivArea,
                     features.MasVnrArea, features.FullBath, features.TotalBsmtSF,
                     features.YearBuilt, features.YearRemodAdd, features.Fireplaces,
                     features.OpenPorchSF, features.LotFrontage, features.WoodDeckSF,
                     features.BsmtFinSF1]

    # Make prediction using the loaded model
    prediction = model.predict([features_list])[0]

    # Return the prediction
    return {'prediction': prediction}
