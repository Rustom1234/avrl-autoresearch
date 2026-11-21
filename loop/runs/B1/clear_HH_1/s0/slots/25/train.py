"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    na = A.sum(axis=1)[:, None]
    nb = B.sum(axis=1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    D = np.array(rows, dtype=np.float64)
    return np.clip(np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def rbf(A, B, gamma):
    sq = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    Tr_tan, Tr_rbf = tanimoto(X, X), rbf(D, D, gamma)
    Te_tan, Te_rbf = tanimoto(Xt, X), rbf(Dt, D, gamma)
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    yn = (y - mu) / sd
    perm = np.random.RandomState(seed).permutation(len(yn))
    inner = np.array_split(perm, 4)
    best_w, best_rho = 0.5, -np.inf
    for w in [0.25, 0.5, 0.75]:
        K = w * Tr_tan + (1.0 - w) * Tr_rbf
        oof = np.zeros(len(yn))
        for va in inner:
            tr = np.setdiff1d(perm, va)
            m = SVR(kernel="precomputed", C=5.0, epsilon=0.05).fit(K[np.ix_(tr, tr)], yn[tr])
            oof[va] = m.predict(K[np.ix_(va, tr)])
        rho = np.corrcoef(rankdata(oof), rankdata(yn))[0, 1]
        if rho > best_rho:
            best_w, best_rho = w, rho
    K_train = best_w * Tr_tan + (1.0 - best_w) * Tr_rbf
    K_test = best_w * Te_tan + (1.0 - best_w) * Te_rbf
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_train, yn)
    return model.predict(K_test)
