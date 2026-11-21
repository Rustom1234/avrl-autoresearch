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
from lightgbm import LGBMRegressor, early_stopping
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def scaffold_holdout(smiles, seed, frac=0.15):
    groups = {}
    for i, s in enumerate(smiles):
        groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(smiles=s), []).append(i)
    keys = list(groups)
    np.random.RandomState(seed).shuffle(keys)
    val = []
    for k in keys:
        if len(val) >= frac * len(smiles):
            break
        val += groups[k]
    return np.array(val)


def tanimoto(A, B):
    # min/max kernel for count vectors
    num = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        num[i] = np.minimum(A[i], B).sum(1) / np.maximum(np.maximum(A[i], B).sum(1), 1e-9)
    return num


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    Xg = xgb_features(train_smiles)
    params = dict(learning_rate=0.03, num_leaves=15, max_depth=4, subsample=0.7, subsample_freq=1,
                  colsample_bytree=0.3, min_child_samples=10, reg_lambda=5, n_jobs=8, random_state=seed,
                  verbose=-1)
    val = scaffold_holdout(train_smiles, seed)
    mask = np.zeros(len(y), dtype=bool)
    mask[val] = True
    m1 = LGBMRegressor(n_estimators=1500, **params)
    m1.fit(Xg[~mask], y[~mask], eval_set=[(Xg[mask], y[mask])], callbacks=[early_stopping(50, verbose=False)])
    best = max(100, int(m1.best_iteration_ * 1.1))
    model = LGBMRegressor(n_estimators=best, **params)
    model.fit(Xg, y)
    p_rf = model.predict(xgb_features(test_smiles))
    svr = SVR(kernel="precomputed", C=5.0, epsilon=0.1)
    svr.fit(tanimoto(Xtr, Xtr), (y - y.mean()) / (y.std() + 1e-9))
    p_svr = svr.predict(tanimoto(Xte, Xtr))
    return rankdata(p_rf) / len(p_rf) + rankdata(p_svr) / len(p_svr)
