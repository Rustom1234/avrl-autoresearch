"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    preds = []
    i = 0
    for max_depth in (4, 6, 8):
        for colsample in (0.2, 0.5):
            # fewer trees for deeper models so each member costs roughly the same
            n_est = int(600 * 6 / max_depth)
            model = XGBRegressor(objective='reg:absoluteerror', tree_method='hist', max_depth=max_depth,
                                 learning_rate=0.03, n_estimators=n_est, subsample=0.8,
                                 colsample_bytree=colsample, min_child_weight=3,
                                 random_state=seed + i, n_jobs=8)
            model.fit(X_train, y)
            preds.append(model.predict(X_test))
            i += 1
    return np.median(np.stack(preds), axis=0)
