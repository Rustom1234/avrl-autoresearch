"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import KFold
from rdkit.Chem import Descriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdReducedGraphs
from xgboost import XGBRegressor
from sklearn.svm import SVR

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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    model = XGBRegressor(max_depth=4, n_estimators=600, learning_rate=0.03, subsample=0.7, colsample_bytree=0.3,
                         min_child_weight=3, reg_lambda=5, n_jobs=8, random_state=seed)
    model.fit(xgb_features(train_smiles), y)
    p_rf = model.predict(xgb_features(test_smiles))
    ys = (y - y.mean()) / (y.std() + 1e-9)
    Ktr = tanimoto(Xtr, Xtr)
    best_c, best_s = 5.0, -np.inf
    for c in [1.0, 2.0, 5.0, 10.0, 20.0]:
        scores = []
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(Ktr):
            m = SVR(kernel="precomputed", C=c, epsilon=0.1).fit(Ktr[np.ix_(tr, tr)], ys[tr])
            scores.append(spearmanr(m.predict(Ktr[np.ix_(va, tr)]), ys[va])[0])
        if np.mean(scores) > best_s:
            best_c, best_s = c, np.mean(scores)
    svr = SVR(kernel="precomputed", C=best_c, epsilon=0.1)
    svr.fit(Ktr, ys)
    p_svr = svr.predict(tanimoto(Xte, Xtr))
    return rankdata(p_rf) / len(p_rf) + rankdata(p_svr) / len(p_svr)
