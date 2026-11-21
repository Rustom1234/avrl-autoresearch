"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import lightgbm as lgb
from scipy.spatial.distance import cdist

_AP_CACHE = {}


def ap_counts(smiles):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    for s in smiles:
        if s not in _AP_CACHE:
            _AP_CACHE[s] = gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32)
    return np.array([_AP_CACHE[s] for s in smiles], dtype=np.float32)


def minmax_kernel(A):
    sa = A.sum(1)
    S = (sa[:, None] + sa[None, :] - cdist(A, A, "cityblock")) / 2
    return S / (sa[:, None] + sa[None, :] - S + 1e-9)

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


GROUPS = [Chem.MolFromSmarts(p) for p in [
    "[CX3](=O)[OX1H0-,OX2H1]",
    "[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N#*);!$(N=*)]([#6])",
    "[NX3][CX3]=[NX2]",
    "S(=O)(=O)[NX3]",
    "c[OX2H]",
    "c1nnn[nH]1",
    "[NX4+]",
    "[nX2;r6]",
]]


_FEAT_CACHE = {}


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        if s in _FEAT_CACHE:
            rows.append(_FEAT_CACHE[s])
            continue
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e5, 1e5).astype(np.float32)
        grp = np.array([len(m.GetSubstructMatches(p)) for p in GROUPS], dtype=np.float32)
        _FEAT_CACHE[s] = np.concatenate([fp, desc, grp])
        rows.append(_FEAT_CACHE[s])
    return np.array(rows, dtype=np.float32)


def crippen(smiles):
    return np.array([Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def tanimoto(A, B):
    a = A @ B.T
    return a / (A.sum(1)[:, None] + B.sum(1)[None, :] - a + 1e-9)


def base_preds(Xtr, ytr, Xte, seed, n_trees, Ktr, Kte):
    et = ExtraTreesRegressor(n_estimators=n_trees, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                             n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    gb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, min_child_samples=10, n_jobs=8, random_state=seed,
                           verbose=-1).fit(Xtr, ytr)
    Ftr, Fte = Xtr[:, :N_BITS], Xte[:, :N_BITS]
    sv = SVR(kernel="precomputed", C=5.0, epsilon=0.1).fit(tanimoto(Ftr, Ftr), ytr)
    sp = SVR(kernel="precomputed", C=5.0, epsilon=0.1).fit(Ktr, ytr)
    return np.column_stack([et.predict(Xte), gb.predict(Xte), sv.predict(tanimoto(Fte, Ftr)), sp.predict(Kte)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    ctr, cte = crippen(train_smiles), crippen(test_smiles)
    y = np.asarray(train_y, dtype=np.float64) - ctr
    K = minmax_kernel(ap_counts(list(train_smiles) + list(test_smiles)))
    n = len(y)
    Ktt, Kte = K[:n, :n], K[n:, :n]
    oof = np.zeros((n, 4))
    for tr, va in KFold(3, shuffle=True, random_state=seed).split(Xtr):
        oof[va] = base_preds(Xtr[tr], y[tr], Xtr[va], seed, 100, Ktt[np.ix_(tr, tr)], Ktt[np.ix_(va, tr)])
    meta = LinearRegression(positive=True, fit_intercept=False).fit(oof, y)
    return meta.predict(base_preds(Xtr, y, Xte, seed, N_TREES, Ktt, Kte)) + cte
