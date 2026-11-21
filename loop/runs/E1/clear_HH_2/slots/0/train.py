"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from rdkit.Chem import Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor

# Settings
N_BITS = 2048
RADIUS = 2
W_DESC = 0.6  # weight of the descriptor-based LightGBM rank; fingerprint ExtraTrees gets 1 - W_DESC


def desc_features(smiles):
    X = np.array([[f(Chem.MolFromSmiles(s)) for _, f in Descriptors.descList] for s in smiles], dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6)


def count_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(train_smiles), y)
    p_desc = lgbm.predict(desc_features(test_smiles))

    et = ExtraTreesRegressor(n_estimators=500, max_features=0.3, min_samples_leaf=1, n_jobs=-1, random_state=seed)
    et.fit(count_features(train_smiles), y)
    p_fp = et.predict(count_features(test_smiles))

    n = len(p_desc)
    return W_DESC * rankdata(p_desc) / n + (1 - W_DESC) * rankdata(p_fp) / n
