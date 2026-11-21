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
    gens = [
        rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS),
        rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=N_BITS,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()),
        rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=N_BITS, maxPath=6),
    ]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [np.array([g.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float64) for g in gens]


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
    Xs = featurize(train_smiles)
    Xts = featurize(test_smiles)
    D = descriptors(train_smiles)
    Dt = descriptors(test_smiles)
    keep = D.std(axis=0) > 1e-9
    D, Dt = D[:, keep], Dt[:, keep]
    dmu, dsd = D.mean(axis=0), D.std(axis=0) + 1e-9
    D = np.clip((D - dmu) / dsd, -5.0, 5.0)
    Dt = np.clip((Dt - dmu) / dsd, -5.0, 5.0)
    gamma = 1.0 / D.shape[1]
    S_train = np.mean([tanimoto(X, X) for X in Xs], axis=0)
    S_test = np.mean([tanimoto(Xt, X) for X, Xt in zip(Xs, Xts)], axis=0)
    K_train = 0.5 * (S_train + rbf(D, D, gamma))
    K_test = 0.5 * (S_test + rbf(Dt, D, gamma))
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    mu, sd = y.mean(), y.std() + 1e-9
    model = SVR(kernel="precomputed", C=5.0, epsilon=0.05)
    model.fit(K_train, (y - mu) / sd)
    return model.predict(K_test)
