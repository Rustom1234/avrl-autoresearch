"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from lightgbm import LGBMRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([d, mg.GetCountFingerprintAsNumPy(m), ap.GetCountFingerprintAsNumPy(m)]))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.2, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
