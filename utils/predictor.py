from settings.constants import Config
from dataloader import Dataloader
import pandas as pd
import joblib


class Predictor:
    def __init__(self):
        self.loader = Dataloader()
        self.loaded_pipeline = joblib.load(Config.SAVED_MODEL)
        self.test_data = pd.read_csv(Config.VAL_CSV)

    def test_model(self):

        X_val = self.loader.preprocess_data(self.test_data)

        test_first_row = self.loaded_pipeline.predict(X_val)

        return test_first_row.astype('int')


if __name__ == '__main__':
    p = Predictor()
    print(p.test_model())
