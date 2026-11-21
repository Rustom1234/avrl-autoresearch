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
from sklearn.linear_model import Ridge
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

PHYS = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumHDonors,
        Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds, Descriptors.RingCount]


def physchem(smiles):
    return np.array([[f(Chem.MolFromSmiles(s)) for f in PHYS] for s in smiles], dtype=np.float64)


def gobbi(smiles, n=4096):
    out = np.zeros((len(smiles), n))
    for i, s in enumerate(smiles):
        fp = Generate.Gen2DFingerprint(Chem.MolFromSmiles(s), Gobbi_Pharm2D.factory)
        for b in fp.GetOnBits():
            out[i, b % n] = 1.0
    return out

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
    G = gobbi(train_smiles)
    Gt = gobbi(test_smiles)
    K_train = (tanimoto(X, X) + rbf(D, D, gamma) + tanimoto(G, G)) / 3.0
    K_test = (tanimoto(Xt, X) + rbf(Dt, D, gamma) + tanimoto(Gt, G)) / 3.0
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    P = physchem(train_smiles)
    Pt = physchem(test_smiles)
    pmu, psd = P.mean(axis=0), P.std(axis=0) + 1e-9
    P = np.clip((P - pmu) / psd, -5.0, 5.0)
    Pt = np.clip((Pt - pmu) / psd, -5.0, 5.0)
    ridge = Ridge(alpha=10.0).fit(P, y)
    r = y - ridge.predict(P)
    mu, sd = r.mean(), r.std() + 1e-9
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_train, (r - mu) / sd)
    return model.predict(K_test) * sd + mu + ridge.predict(Pt)
