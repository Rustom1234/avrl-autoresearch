"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata
from rdkit.Chem import Descriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdReducedGraphs
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def xgb_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=1024)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = [f(m) for _, f in Descriptors.descList]
        av = np.array(list(pyAvalonTools.GetAvalonCountFP(m, nBits=1024)), dtype=np.float64)
        erg = np.asarray(rdReducedGraphs.GetErGFingerprint(m), dtype=np.float64)
        rows.append(np.concatenate([d, gen.GetCountFingerprintAsNumPy(m), av, erg]))
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto(A, B):
    # min/max kernel for count vectors
    num = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        num[i] = np.minimum(A[i], B).sum(1) / np.maximum(np.maximum(A[i], B).sum(1), 1e-9)
    return num


def svr_fit_predict(K_tr, y_tr, K_te):
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(K_tr, (y_tr - y_tr.mean()) / (y_tr.std() + 1e-9))
    return svr.predict(K_te)


def xgb_model(n_trees, seed):
    return XGBRegressor(max_depth=4, n_estimators=n_trees, learning_rate=0.03, subsample=0.7, colsample_bytree=0.3,
                        min_child_weight=3, reg_lambda=5, n_jobs=8, random_state=seed)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    Gtr, Gte = xgb_features(train_smiles), xgb_features(test_smiles)
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    Ktr = tanimoto(Xtr, Xtr)
    Kte = tanimoto(Xte, Xtr)
    # inner 5-fold out-of-fold predictions for both members, used to choose the blend weight
    oof_x = np.zeros(len(y))
    oof_s = np.zeros(len(y))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(y):
        oof_x[b] = xgb_model(300, seed).fit(Gtr[a], y[a]).predict(Gtr[b])
        oof_s[b] = svr_fit_predict(Ktr[np.ix_(a, a)], y[a], Ktr[np.ix_(b, a)])
    rx = rankdata(oof_x) / len(y)
    rs = rankdata(oof_s) / len(y)
    best_w, best_r = 0.5, -np.inf
    for w in np.arange(0, 1.01, 0.1):
        r = np.corrcoef(rankdata(w * rs + (1 - w) * rx), y)[0, 1]
        if r > best_r:
            best_w, best_r = w, r
    # final members on the full training set
    p_rf = xgb_model(600, seed).fit(Gtr, y).predict(Gte)
    p_svr = svr_fit_predict(Ktr, y, Kte)
    return best_w * rankdata(p_svr) / len(p_svr) + (1 - best_w) * rankdata(p_rf) / len(p_rf)
