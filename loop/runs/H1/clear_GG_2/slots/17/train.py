"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Settings
N_BITS = 2048
RADIUS = 3
N_TREES_RF = 300
N_TREES_XGB = 300
N_TREES_LGBM = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.log1p(np.asarray(train_y, dtype=np.float64))

    rf = RandomForestRegressor(n_estimators=N_TREES_RF, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    
    xgb = XGBRegressor(n_estimators=N_TREES_XGB, max_depth=6, learning_rate=0.1, random_state=seed)
    xgb.fit(X_train, y_train)
    
    lgbm = LGBMRegressor(n_estimators=N_TREES_LGBM, max_depth=6, learning_rate=0.1, random_state=seed)
    lgbm.fit(X_train, y_train)
    
    preds = (rf.predict(X_test) + xgb.predict(X_test) + lgbm.predict(X_test)) / 3
    return np.expm1(preds)
