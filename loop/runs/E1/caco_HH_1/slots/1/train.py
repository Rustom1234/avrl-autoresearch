"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        m = Chem.MolFromSmiles(smi)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = np.concatenate([fp, desc])
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    # Stage 1: random forest on Morgan counts, with out-of-bag predictions on the training set
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               bootstrap=True, oob_score=True, random_state=seed, n_jobs=8)
    rf.fit(X_tr[:, :N_BITS], y)
    resid = y - rf.oob_prediction_
    rf_te = rf.predict(X_te[:, :N_BITS])
    # Stage 2: L1 LightGBM on RDKit descriptors predicts the OOB residuals
    gbm = LGBMRegressor(objective='l1', n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.3, min_child_samples=10, random_state=seed,
                        n_jobs=8, verbose=-1)
    gbm.fit(X_tr[:, N_BITS:], resid)
    return rf_te + gbm.predict(X_te[:, N_BITS:])
