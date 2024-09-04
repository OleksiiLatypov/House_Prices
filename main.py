from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
import os
from datetime import datetime

from settings.constants import Config
from utils.dataloader import loader
from utils.upload_file import Uploader

# Initialize constants
# PROCESSED_DIR = "processed_files"
# os.makedirs(PROCESSED_DIR, exist_ok=True)
# current_year = datetime.now().year


class HouseFeatures(BaseModel):
    OverallQual: Optional[int] = Field(6, ge=0, le=10)
    GrLivArea: Optional[int] = Field(1460, ge=1)
    GarageCars: Optional[int] = Field(2, ge=0, le=10)
    TotalBsmtSF: Optional[float] = Field(1000, ge=0)
    FullBath: Optional[int] = Field(2, ge=0, le=5)
    YearBuilt: Optional[int] = Field(1973, ge=1872, le=Config.current_year)
    YearRemodAdd: Optional[int] = Field(1994, ge=1950, le=Config.current_year)
    MasVnrArea: Optional[int] = Field(0, ge=0)
    Fireplaces: Optional[int] = Field(2, ge=0)
    HeatingQC: Optional[str] = Field('Gd', pattern='^(Ex|Gd|TA|Fa|Po)$')
    KitchenQual: Optional[str] = Field('Gd', pattern='^(Ex|Gd|TA|Fa|Po)$')

    @validator('YearRemodAdd')
    def check_year_remod_add(cls, value, values):
        if 'YearBuilt' in values and value < values['YearBuilt']:
            raise ValueError('YearRemodAdd must be greater than or equal to YearBuilt')
        return value


# Initialize FastAPI app
app = FastAPI()

# Load machine learning model

try:
    loaded_pipeline = joblib.load(Config.SAVED_MODEL)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.get('/api/healthchecker')
def healthchecker():
    return {'message': 'Welcome'}


@app.post('/api/predict/')
async def predict(features: HouseFeatures):
    """
    Endpoint to make predictions using the loaded ML model.
    Expects a JSON payload with the features required for prediction.
    """
    try:
        features_dict = {k: [getattr(features, k)] for k in features.__fields__.keys()}
        data_to_predict = pd.DataFrame(features_dict)
        data_to_predict = loader.preprocess_for_prediction(data_to_predict)
        prediction = loaded_pipeline.predict(data_to_predict)[0]
        return {"prediction SalePrice": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/api/predict_from_csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Endpoint to make predictions using the loaded ML model from a CSV file.
    Expects a CSV file upload containing the features required for prediction.
    """
    X = Uploader.file_uploader(file.file)
    try:
        csv_prediction = loaded_pipeline.predict(X)
        return {"predictions": csv_prediction.astype("int").tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed, {e}")


@app.post('/api/upload_and_preprocess')
async def upload_for_predictions(file: UploadFile = File(...)):
    """
    Endpoint to upload a CSV file, preprocess the data, and return predictions.
    The result is returned as a downloadable CSV file.
    """
    X = Uploader.file_uploader(file.file)
    try:
        X_predictions = loaded_pipeline.predict(X)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed, {e}")
    X['predictions'] = X_predictions
    try:
        processed_filename = f"{Config.PROCESSED_DIR}/saleprice_predictions_for_{file.filename}"
        X.to_csv(processed_filename, index=False)

        return StreamingResponse(
            open(processed_filename, "rb"),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={os.path.basename(processed_filename)}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to preprocess: {e}")


@app.get("/download/{filename}")
async def download_file_with_predictions(filename: str):
    """
    Endpoint to download a CSV file containing predictions.
    """
    file_path = os.path.join(Config.PROCESSED_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)
