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
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed).fit(Xtr, y)
    Dtr = np.hstack([descriptors(train_smiles), Xtr])
    Dte = np.hstack([descriptors(test_smiles), Xte])
    gb_preds = []
    for k in range(5):
        gbk = lgb.LGBMRegressor(objective="huber", n_estimators=400, learning_rate=0.03, num_leaves=8,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.3, min_child_samples=5,
                                random_state=seed * 10 + k, n_jobs=8, verbose=-1).fit(Dtr, y)
        gb_preds.append(gbk.predict(Dte))
    gb_pred = np.mean(gb_preds, axis=0)
    svr = SVR(kernel="precomputed", C=5.0).fit(tanimoto(Xtr, Xtr), y)
    preds = [rf.predict(Xte), gb_pred, svr.predict(tanimoto(Xte, Xtr))]
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)
