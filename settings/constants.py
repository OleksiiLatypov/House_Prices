import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
#print(ROOT_DIR)

DATA_DIR = 'data'
MODELS_DIR = 'models'

TRAIN_CSV = os.path.join(ROOT_DIR, DATA_DIR, 'train.csv')
VAL_CSV = os.path.join(ROOT_DIR, DATA_DIR, 'test.csv')
SAVED_ESTIMATOR = os.path.join(ROOT_DIR, MODELS_DIR, 'CatBoostClassififer.pkl')
SAVED_SCALER = os.path.join(ROOT_DIR, MODELS_DIR, 'scaler.pkl')

# print(TRAIN_CSV)
# print(VAL_CSV)
print(SAVED_ESTIMATOR)

