"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    Mtr, Mte = mixed_features(train_smiles), mixed_features(test_smiles)
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed).fit(Mtr, y)
    gb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=8, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.2, min_child_samples=10, reg_lambda=5, random_state=seed, n_jobs=8,
                           verbose=-1).fit(Mtr, y)
    svr = SVR(kernel="precomputed", C=5.0).fit(tanimoto(Xtr, Xtr), y)
    preds = [rf.predict(Mte), gb.predict(Mte), svr.predict(tanimoto(Xte, Xtr))]
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)
