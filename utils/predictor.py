from settings.constants import TRAIN_CSV, SAVED_SCALER, VAL_CSV, SAVED_MODEL
from dataloader import Dataloader
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np


class Predictor:
    def __init__(self):
        self.loader = Dataloader()
        self.loaded_pipeline = joblib.load(SAVED_MODEL)

    def test_model(self):

        X_val = self.loader.preprocess_data()

        test_first_row = self.loaded_pipeline.predict(X_val)

        return test_first_row.astype('int')


if __name__ == '__main__':
    p = Predictor()
    print(p.test_model())
