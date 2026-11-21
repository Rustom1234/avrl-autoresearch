"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        cnt = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, cnt]))
    X = np.nan_to_num(np.array(rows), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    models = [
        RandomForestRegressor(n_estimators=300, max_features=0.3, n_jobs=8, random_state=seed),
        LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.4, reg_lambda=5, n_jobs=8,
                      random_state=seed, verbose=-1),
        CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, thread_count=8,
                          random_seed=seed, verbose=0),
    ]
    preds = [m.fit(X, y).predict(Xt) for m in models]
    return np.mean(preds, axis=0)
