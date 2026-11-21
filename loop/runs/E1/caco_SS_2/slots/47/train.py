"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics.pairwise import rbf_kernel

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def morgan_counts(smiles):
    return np.array([_GEN.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax_kernel(A, B):
    K = np.empty((len(A), len(B)), dtype=np.float64)
    for i, a in enumerate(A):
        K[i] = np.minimum(a, B).sum(1) / np.maximum(np.maximum(a, B).sum(1), 1e-9)
    return K

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
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    model.fit(Xtr, y)
    p_lgb = model.predict(Xte)
    keep = Xtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
    Str = qt.fit_transform(Xtr[:, keep])
    Ste = qt.transform(Xte[:, keep])
    gamma = 1.0 / (Str.shape[1] * Str.var())
    Mtr, Mte = morgan_counts(train_smiles), morgan_counts(test_smiles)
    Ktr = 0.6 * rbf_kernel(Str, Str, gamma=gamma) + 0.4 * minmax_kernel(Mtr, Mtr)
    Kte = 0.6 * rbf_kernel(Ste, Str, gamma=gamma) + 0.4 * minmax_kernel(Mte, Mtr)
    svr = SVR(kernel='precomputed', C=3, epsilon=0.1).fit(Ktr, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Kte)
