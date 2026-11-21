"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from xgboost import XGBRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8,
                         colsample_bytree=0.5, objective="reg:absoluteerror", random_state=seed, n_jobs=8)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
