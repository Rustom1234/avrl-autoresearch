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
RADIUS = 2
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
    
    # Simple hold-out for XGBoost early stopping
    train_idx = np.split(np.arange(len(X_train)), [int(0.8 * len(X_train))])[0]
    val_idx = np.split(np.arange(len(X_train)), [int(0.8 * len(X_train))])[1]
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    xgb = XGBRegressor(n_estimators=N_TREES_XGB, max_depth=6, learning_rate=0.1, random_state=seed)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    lgbm = LGBMRegressor(n_estimators=N_TREES_LGBM, max_depth=6, learning_rate=0.1, random_state=seed, verbose=-1)
    lgbm.fit(X_train, y_train)
    
    preds = (rf.predict(X_test) + xgb.predict(X_test) + lgbm.predict(X_test)) / 3
    return np.expm1(preds)
