"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


N_DESC = 100


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        r = []
        for _, f in Descriptors.descList:
            try:
                r.append(f(m))
            except Exception:
                r.append(0.0)
        rows.append(r)
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def minmax_kernel(A, B):
    sa, sb = A.sum(1), B.sum(1)
    K = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        mn = np.minimum(A[i], B).sum(1)
        K[i] = mn / np.maximum(sa[i] + sb - mn, 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    import lightgbm as lgb
    y = np.asarray(train_y, dtype=np.float64)
    y = (y - y.mean()) / (y.std() + 1e-9)
    Xtr = np.hstack([descriptors(train_smiles), featurize(train_smiles)])
    Xte = np.hstack([descriptors(test_smiles), featurize(test_smiles)])
    model = lgb.LGBMRegressor(objective="huber", alpha=1.0, n_estimators=500, learning_rate=0.03,
                              num_leaves=15, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                              min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    model.fit(Xtr, y)
    return model.predict(Xte)
