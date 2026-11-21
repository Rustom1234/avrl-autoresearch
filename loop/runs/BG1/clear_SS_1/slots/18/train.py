"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import rankdata
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def mixed_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = [f(m) for _, f in Descriptors.descList]
        rows.append(np.concatenate([np.array(d, dtype=np.float64), gen.GetCountFingerprintAsNumPy(m)]))
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def count_fp(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax(A, B, chunk=16):
    sa, sb = A.sum(1), B.sum(1)
    out = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(0, len(A), chunk):
        inter = np.minimum(A[i:i + chunk, None, :], B[None, :, :]).sum(2)
        out[i:i + chunk] = inter / (sa[i:i + chunk, None] + sb[None, :] - inter + 1e-9)
    return out


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed).fit(mixed_features(train_smiles), y)
    gb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, min_child_samples=5, random_state=seed, n_jobs=8,
                           verbose=-1).fit(Xtr, y)
    svr = SVR(kernel="precomputed", C=5.0).fit(tanimoto(Xtr, Xtr), y)
    Ctr, Cte = count_fp(train_smiles), count_fp(test_smiles)
    kr = KernelRidge(kernel="precomputed", alpha=1.0).fit(minmax(Ctr, Ctr), y - y.mean())
    preds = [rf.predict(mixed_features(test_smiles)), gb.predict(Xte), svr.predict(tanimoto(Xte, Xtr)),
             kr.predict(minmax(Cte, Ctr))]
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)
