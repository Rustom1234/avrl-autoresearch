"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

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
    return np.array(rows, dtype=np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = descriptors(train_smiles)
    Xt = descriptors(test_smiles)
    X = np.where(np.isfinite(X), X, np.nan)
    Xt = np.where(np.isfinite(Xt), Xt, np.nan)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)
    Xt = np.where(np.isnan(Xt), med, Xt)
    sc = StandardScaler().fit(X)
    X = np.clip(sc.transform(X), -10, 10)
    Xt = np.clip(sc.transform(Xt), -10, 10)
    y = np.asarray(train_y, dtype=np.float64)
    y = (rankdata(y) - 0.5) / len(y)
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(X, y)
    return model.predict(Xt)
