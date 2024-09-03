from fastapi import FastAPI, UploadFile, File, HTTPException, Path, Query, Depends
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pydantic import BaseModel, PositiveInt, Field, PositiveFloat, validator, field_validator
import pandas as pd
import pickle

from settings.constants import SAVED_MODEL
from utils.dataloader import Dataloader
from catboost import CatBoostRegressor
from datetime import datetime
# from db import get_db, Prediction
# from sqlalchemy.orm import Session
# from sqlalchemy import text
import joblib
import io
import os

current_year = datetime.now().year
# print(current_year)
# Define a Pydantic model to validate the request body
lst = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces', 'BsmtFinSF1',
       'LotFrontage', 'HeatingQC', 'KitchenQual']


class HouseFeatures(BaseModel):
    OverallQual: Optional[int] = Field(6, ge=0, le=10)
    GrLivArea: PositiveInt
    GarageCars: Optional[int] = Field(2, ge=0, le=10)
    TotalBsmtSF: Optional[float] = Field(1000, ge=0)
    FullBath: Optional[int] = Field(2, ge=0, le=5)
    YearBuilt: Optional[int] = Field(1973, ge=1872, le=current_year)
    YearRemodAdd: Optional[int] = Field(1994, ge=1950, le=current_year)
    MasVnrArea: Optional[int] = Field(0, ge=0)
    Fireplaces: Optional[int] = Field(2, ge=0)
    BsmtFinSF1: Optional[int] = Field(None, ge=0)
    LotFrontage: Optional[int] = Field(None, ge=0)
    HeatingQC: Optional[str] = Field('Gd')
    KitchenQual: Optional[str] = Field('Gd')

    @field_validator('YearRemodAdd')
    def check_year_remod_add(cls, value, info):
        print(info)
        if value < info.data['YearBuilt']:
            raise ValueError('YearRemodAdd must be greater than or equal to YearBuilt')
        return value


app = FastAPI()

# Load your machine learning model from the pickle file
loaded_pipeline = joblib.load(SAVED_MODEL)

print(loaded_pipeline)


@app.get('/api/healthchecker')
def healthchecker():
    return {'message': 'Welcome'}


@app.post('/api/predict')
async def predict(features: HouseFeatures):
    """
    Endpoint to make predictions using the loaded ML model.
    Expects a JSON payload with the features required for prediction.
    """
    loader = Dataloader()
    # Extract features from the request data and format them for prediction
    # features_list = [features.OverallQual, features.GrLivArea, features.GarageCars,
    #                  features.TotalBsmtSF, features.FullBath, features.YearBuilt,
    #                  features.YearRemodAdd, features.MasVnrArea, features.Fireplaces, features.BsmtFinSF1,
    #                  features.LotFrontage, features.HeatingQC, features.KitchenQual]
    features_dict = {
        'OverallQual': [features.OverallQual],
        'GrLivArea': [features.GrLivArea],
        'GarageCars': [features.GarageCars],
        'TotalBsmtSF': [features.TotalBsmtSF],
        'FullBath': [features.FullBath],
        'YearBuilt': [features.YearBuilt],
        'YearRemodAdd': [features.YearRemodAdd],
        'MasVnrArea': [features.MasVnrArea],
        'Fireplaces': [features.Fireplaces],
        'BsmtFinSF1': [features.BsmtFinSF1],
        'LotFrontage': [features.LotFrontage],
        'HeatingQC': [features.HeatingQC],
        'KitchenQual': [features.KitchenQual]
    }
    # Make prediction using the loaded model
    data_to_predict = pd.DataFrame(features_dict)
    print(features_dict)
    data_to_predict = loader.preprocess_for_prediction(data_to_predict)
    print(data_to_predict)
    prediction = loaded_pipeline.predict(data_to_predict)[0]

    # Return the prediction
    return {'prediction': prediction}


@app.post("/api/predict_from_csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Endpoint to make predictions using the loaded ML model from a CSV file.
    Expects a CSV file upload containing the features required for prediction.
    """
    # Read the CSV file
    # df = pd.read_csv(file.file)
    loader = Dataloader()
    # preprocessed_df = loader.load()
    # print(preprocessed_df)
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail='Error reading csv file')
    X = loader.preprocess_data(df)
    csv_prediction = loaded_pipeline.predict(X)
    print(csv_prediction)
    return {"predictions": csv_prediction.tolist()}


PROCESSED_DIR = "processed_files"
os.makedirs(PROCESSED_DIR, exist_ok=True)


@app.post('/api/upload_and_preprocess')
async def upload_for_predictions(file: UploadFile = File(...)):
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(file.file)

    # Preprocess the data
    loader = Dataloader()
    X = loader.preprocess_data(df)

    # Make predictions
    X_predictions = loaded_pipeline.predict(X)

    # Add predictions to the original DataFrame
    df['predictions'] = X_predictions

    # Save the result to a new CSV file
    processed_filename = f"{PROCESSED_DIR}/saleprice_predictions_for_{file.filename}"
    df.to_csv(processed_filename, index=False)

    # Return a response with a success message and the file download link
    return {
        "message": "File uploaded and preprocessed successfully!",
        "download_link": f"/download/{os.path.basename(processed_filename)}"
    }


@app.get("/download/{filename}")
async def download_file_with_predictions(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)
