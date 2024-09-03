import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from settings.constants import TRAIN_CSV, VAL_CSV
import joblib
import numpy as np


# df = pd.read_csv(TRAIN_CSV)


class Dataloader:
    def __init__(self):
        self.dataset = pd.read_csv(TRAIN_CSV)
        self.test_data = pd.read_csv(VAL_CSV)

    def fill_miss_values(self, numeric_columns, dataset):
        for column in numeric_columns:
            column_mean_value = dataset[column].mean()
            dataset.loc[:, column] = dataset[column].fillna(column_mean_value)
        return dataset

    def fill_miss_cat(self, dataset):
        cat_columns = dataset.select_dtypes(include='object')
        for column in cat_columns:
            column_mode_value = dataset[column].mode()[0]  # Access the first element of the mode
            dataset.loc[:, column] = dataset[column].fillna(column_mode_value)
        return dataset

    def map_feature(self, dataset):
        important_cat = ["HeatingQC", "KitchenQual"]
        dataset = dataset[important_cat]
        dataset = self.fill_miss_cat(dataset)
        heatingqc_mapping = {
            'Ex': 4,
            'Gd': 3,
            'TA': 2,
            'Fa': 1,
            'Po': 0
        }
        # Apply the mapping to the HeatingQC column
        dataset.loc[:, 'HeatingQC'] = dataset['HeatingQC'].map(heatingqc_mapping)
        kitchenqual_mapping = {
            'Ex': 4,
            'Gd': 3,
            'TA': 2,
            'Fa': 1,
            'Po': 0
        }
        # # Apply the mapping to the KitchenQual column
        dataset.loc[:, 'KitchenQual'] = dataset['KitchenQual'].map(kitchenqual_mapping)
        return dataset

    def load(self):
        numeric_columns = self.dataset.select_dtypes(include='number')
        correlation_matrix = numeric_columns.corr()
        correlation_with_target = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)

        highly_correlated = [column for column in correlation_with_target.index if
                             correlation_with_target[column] > 0.35]
        columns_to_drop = ['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd', 'GarageYrBlt']
        important_cat = self.map_feature(self.dataset)
        X = pd.concat([self.dataset[highly_correlated], important_cat], axis=1)
        X = self.fill_miss_values(highly_correlated, X)
        X = X.drop(columns_to_drop, axis=1)
        y = X['SalePrice']
        X = X.drop('SalePrice', axis=1)
        return X, y

    def preprocess_data(self):
        X, y = self.load()
        important_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt',
                             'YearRemodAdd', 'MasVnrArea', 'Fireplaces']
        catecorical_features = self.map_feature(self.test_data)
        numeric_columns = self.test_data[important_columns].select_dtypes(include='number')
        fill_miss_vals = self.fill_miss_values(numeric_columns, self.test_data[important_columns])
        test = pd.concat([fill_miss_vals, catecorical_features], axis=1)
        print(test.shape)
        return test

    def preprocess_for_prediction(self, features):
        heatingqc_mapping = {
            'Ex': 4,
            'Gd': 3,
            'TA': 2,
            'Fa': 1,
            'Po': 0
        }
        kitchenqual_mapping = {
            'Ex': 4,
            'Gd': 3,
            'TA': 2,
            'Fa': 1,
            'Po': 0
        }
        #data = pd.DataFrame([features.dict()])
        features.loc[:, 'HeatingQC'] = features['HeatingQC'].map(heatingqc_mapping)
        features.loc[:, 'KitchenQual'] = features['KitchenQual'].map(kitchenqual_mapping)
        return features


if __name__ == '__main__':
    loader = Dataloader()
    # test = pd.read_csv(VAL_CSV)
    print(loader.preprocess_data())
    print(loader.load())
