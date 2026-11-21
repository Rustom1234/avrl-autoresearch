"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        m = Chem.MolFromSmiles(smi)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = np.concatenate([fp, desc])
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_tr = featurize(train_smiles)[:, N_BITS:].astype(np.float64)
    X_te = featurize(test_smiles)[:, N_BITS:].astype(np.float64)
    keep = X_tr.std(axis=0) > 0
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]
    qt = QuantileTransformer(n_quantiles=len(X_tr), output_distribution='normal', random_state=seed)
    X_tr = qt.fit_transform(X_tr)
    X_te = qt.transform(X_te)
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    gp.fit(X_tr, np.asarray(train_y, dtype=np.float64))
    return gp.predict(X_te)
