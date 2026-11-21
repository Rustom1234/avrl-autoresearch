"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

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


def morgan_counts(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax_kernel(A, B):
    sa, sb = A.sum(1), B.sum(1)
    K = np.empty((len(A), len(B)))
    for j in range(len(B)):
        mn = np.minimum(A, B[j]).sum(1)
        K[:, j] = mn / np.maximum(sa + sb[j] - mn, 1e-9)
    return K


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
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    Ftr, Fte = morgan_counts(train_smiles), morgan_counts(test_smiles)
    tsvr = SVR(kernel='precomputed', C=3, epsilon=0.1).fit(minmax_kernel(Ftr, Ftr), y)
    p_tan = tsvr.predict(minmax_kernel(Ftr, Fte).T)
    return np.median(np.vstack([p_lgb, svr.predict(Ste), p_tan]), axis=0)
