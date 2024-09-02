from typing import Optional, Union

from settings.constants import TRAIN_CSV, SAVED_MODEL
from dataloader import Dataloader
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, ShuffleSplit, RepeatedKFold
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
import numpy as np


# from xgboost import XGBRegressor


class Estimator:
    def __init__(self, train_csv: Optional[Union[str, pd.DataFrame]] = None,
                 loader=None, scaler=None, model=None) -> None:
        self.train_csv = train_csv if train_csv else pd.read_csv(TRAIN_CSV)
        self.loader = loader if loader else Dataloader()
        self.scaler = scaler if scaler else MinMaxScaler()
        self.model = model
        if self.model is None:
            self.model = CatBoostRegressor(iterations=1150,
                                           learning_rate=0.03,
                                           depth=5,
                                           l2_leaf_reg=1,
                                           loss_function='RMSE',
                                           # bagging_temperature=0.9,
                                           border_count=42,
                                           random_seed=0)

    def train(self):
        X, y = self.loader.load()

        pipeline = Pipeline([('scaler', self.scaler), ('model', self.model)])

        kf = KFold(n_splits=5, shuffle=True, random_state=49)

        cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=make_scorer(mean_squared_error))

        rmse_scores = np.sqrt(cv_scores)

        mean_rmse = np.mean(rmse_scores)

        pipeline.fit(X, y)

        joblib.dump(pipeline, SAVED_MODEL)

        print(rmse_scores)
        print(mean_rmse)
        print(X.shape)
        return 'Finish'


if __name__ == '__main__':
    e = Estimator()
    print(e.train())

    # pipeline = Pipeline([('scaler', self.scaler), ('rf', self.model)])

    # # Define k-fold cross-validation
    # kf = KFold(n_splits=3, shuffle=True, random_state=49)

    # # Use cross-validation to evaluate the model
    # cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring=make_scorer(mean_squared_error))

    # # Calculate RMSE for each fold and the mean RMSE
    # rmse_scores = np.sqrt(cv_scores)
    # mean_rmse = np.mean(rmse_scores)

    # # Fit the pipeline on the entire dataset
    # pipeline.fit(X, y)

    # # Save the trained pipeline to a file
    # joblib.dump(pipeline, '/workspaces/105818016/house_price/models/trained_pipeline.pkl')
    # print(X.shape, y.shape)
    # return f"RMSE: {mean_rmse}"
