"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
FEATURE_FRACTION = 0.8
BAG_FRACTION = 0.8


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    
    # LightGBM
    m1 = LGBMRegressor(n_estimators=N_TREES, feature_fraction=FEATURE_FRACTION, 
                         bag_fraction=BAG_FRACTION, random_state=seed, n_jobs=-1, verbosity=-1)
    m1.fit(X_train, y)
    
    # XGBoost
    m2 = XGBRegressor(n_estimators=N_TREES, colsample_bytree=FEATURE_FRACTION, 
                        subsample=BAG_FRACTION, random_state=seed, n_jobs=-1)
    m2.fit(X_train, y)
    
    # CatBoost
    m3 = CatBoostRegressor(iterations=N_TREES, learning_rate=0.1, 
                            logging_level='Silent', random_seed=seed)
    m3.fit(X_train, y)
    
    return (m1.predict(X_test) + m2.predict(X_test) + m3.predict(X_test)) / 3
