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
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax_kernel(A, B, chunk=8):
    sa, sb = A.sum(1), B.sum(1)
    K = np.zeros((len(A), len(B)), dtype=np.float64)
    for i in range(0, len(A), chunk):
        mn = np.minimum(A[i:i + chunk, None, :], B[None, :, :]).sum(-1)
        K[i:i + chunk] = mn / np.maximum(sa[i:i + chunk, None] + sb[None, :] - mn, 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    svr = SVR(kernel='precomputed', C=5, epsilon=0.1).fit(minmax_kernel(Xtr, Xtr), y)
    return svr.predict(minmax_kernel(Xte, Xtr))
