import os
import sys
from datetime import datetime
from pathlib import Path


class Config:
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    TRAIN_CSV = os.path.join(ROOT_DIR, DATA_DIR, 'train.csv')
    VAL_CSV = os.path.join(ROOT_DIR, DATA_DIR, 'test.csv')
    SAVED_MODEL = os.path.join(ROOT_DIR, MODELS_DIR, 'CatBoostClassififer.pkl')
    PROCESSED_DIR = "processed_files"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    current_year = datetime.now().year

# print(TRAIN_CSV)
# print(VAL_CSV)
# print(SAVED_ESTIMATOR)
