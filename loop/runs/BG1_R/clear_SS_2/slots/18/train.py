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


def featurize(smiles, radius=RADIUS):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=N_BITS)
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
    from sklearn.kernel_ridge import KernelRidge
    from scipy.stats import rankdata
    y = np.asarray(train_y, dtype=np.float64)
    y = rankdata(y) / len(y)
    Ktr, Kte = 0.0, 0.0
    for r in (1, 2, 3):
        Xtr, Xte = featurize(train_smiles, r), featurize(test_smiles, r)
        Ktr = Ktr + minmax_kernel(Xtr, Xtr) / 3
        Kte = Kte + minmax_kernel(Xte, Xtr) / 3
    model = KernelRidge(alpha=1.0, kernel="precomputed")
    model.fit(Ktr, y - y.mean())
    return model.predict(Kte) + y.mean()
