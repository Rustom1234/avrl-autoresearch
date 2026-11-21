"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata, norm
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


def scalings(D, Dt):
    # Three descriptor scalings: z-score (clipped), robust median/IQR (clipped), rank-Gaussian.
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    z = (np.clip((D - dmu) / dsd, -5.0, 5.0), np.clip((Dt - dmu) / dsd, -5.0, 5.0))
    med = np.median(D, axis=0)
    iqr = np.percentile(D, 75, axis=0) - np.percentile(D, 25, axis=0)
    iqr = np.where(iqr > 1e-9, iqr, dsd)
    rb = (np.clip((D - med) / iqr, -5.0, 5.0), np.clip((Dt - med) / iqr, -5.0, 5.0))
    n = D.shape[0]
    q = (np.arange(n) + 0.5) / n
    Ztr = np.empty_like(D)
    Zte = np.empty_like(Dt)
    for j in range(D.shape[1]):
        Ztr[:, j] = norm.ppf((rankdata(D[:, j]) - 0.5) / n)
        Zte[:, j] = norm.ppf(np.interp(Dt[:, j], np.sort(D[:, j]), q))
    return [z, rb, (Ztr, Zte)]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles).astype(np.float64)
    Xt = featurize(test_smiles).astype(np.float64)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    K_tan = tanimoto(X, X)
    Kt_tan = tanimoto(Xt, X)
    gamma = 1.0 / D.shape[1]
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    ys = (y - mu) / sd
    preds = []
    for D_tr, D_te in scalings(D, Dt):
        K_train = 0.5 * (K_tan + rbf(D_tr, D_tr, gamma))
        K_test = 0.5 * (Kt_tan + rbf(D_te, D_tr, gamma))
        model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
        model.fit(K_train, ys)
        preds.append(rankdata(model.predict(K_test)) / len(test_smiles))
    return np.mean(preds, axis=0)
