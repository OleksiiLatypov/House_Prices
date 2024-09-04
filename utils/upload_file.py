import pandas as pd
from fastapi import HTTPException
from typing_extensions import BinaryIO

from utils.dataloader import loader


class Uploader:
    @staticmethod
    def file_uploader(file: BinaryIO) -> pd.DataFrame:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing CSV file: {e}")
        try:
            X = loader.preprocess_data(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Make sure your csv file contains next columns, OverallQual,\
                                                        GrLivArea, GarageCars, TotalBsmtSF, FullBath,\
                                                        YearBuilt, YearRemodAdd, MasVnrArea, Fireplaces,\
                                                        HeatingQC, KitchenQual")
        return X
