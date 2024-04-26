import io

import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pickle
import json

from starlette.responses import StreamingResponse


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
    print(features_list)
    print(type(features_list))
    # Make prediction using the loaded model
    prediction = model.predict([features_list])[0]
    # Return the prediction
    return {'prediction': prediction}


@app.post('/api/upload_predict')
async def predict_dataframe(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    print(df.head())
    indexes = df.iloc[:, 0]
    csv_prediction = model.predict(df)
    print(csv_prediction)
    d = {}
    for index, value in zip(indexes, csv_prediction):
        d[index] = value
    print(d)
    return {'message': 'Data loaded successfully!', 'pred': csv_prediction.tolist(), 'd': d}


@app.get('/api/hyperparameters')
async def get_params():
    hyperparameters = model.get_params()
    return {'Hyperparameters of model': hyperparameters}


@app.post('/api/upload_and_preprocess')
async def upload_and_preprocess(file: UploadFile = File(...)):
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(io.BytesIO(await file.read()))

    # Preprocess the dataset (e.g., perform cleaning, feature engineering, etc.)
    # Example preprocessing steps:
    # - Remove missing values
    # - Encode categorical variables
    # - Normalize numerical variables
    # - Perform feature engineering

    # For demonstration purposes, let's assume we just want to drop missing values
    df.dropna(inplace=True)

    # Save the preprocessed dataset to a new CSV file
    preprocessed_filename = "preprocessed_dataset.csv"
    df.to_csv(preprocessed_filename, index=False)

    return {'message': 'Dataset uploaded, preprocessed, and saved successfully!', 'filename': preprocessed_filename}


@app.get('/api/download_preprocessed_dataset')
async def download_preprocessed_dataset(filename: str):
    # Return the preprocessed dataset file as a response
    content = open(filename, "rb").read()
    return StreamingResponse(io.BytesIO(content), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={filename}"})
