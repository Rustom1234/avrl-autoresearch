"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from scipy.optimize import nnls
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles, kind):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    if kind == "atompair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
        return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    return np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])


N_INNER = 4


def tanimoto(A, B):
    inter = A @ B.T
    na = A.sum(axis=1)[:, None]
    nb = B.sum(axis=1)[None, :]
    return inter / np.maximum(na + nb - inter, 1e-9)


def base_predict(Xtr, ytr, Xte, seed):
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
    p_rf = rf.fit(Xtr, ytr).predict(Xte)
    mu = ytr.mean()
    kr = KernelRidge(alpha=1.0, kernel="precomputed").fit(tanimoto(Xtr, Xtr), ytr - mu)
    p_krr = kr.predict(tanimoto(Xte, Xtr)) + mu
    lg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.3, random_state=seed,
                           verbose=-1, n_jobs=-1)
    p_lgb = lg.fit(Xtr, ytr).predict(Xte)
    return np.column_stack([p_rf, p_krr, p_lgb])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    Xtr = featurize(train_smiles, "morgan")
    Xte = featurize(test_smiles, "morgan")
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    gid = np.unique(np.array(scaffolds, dtype=object), return_inverse=True)[1]
    oof = np.zeros((len(y), 3))
    for tr, va in GroupKFold(n_splits=N_INNER).split(Xtr, y, gid):
        oof[va] = base_predict(Xtr[tr], y[tr], Xtr[va], seed)
    w, _ = nnls(oof, y)
    if w.sum() <= 0:
        w = np.ones(3) / 3
    return base_predict(Xtr, y, Xte, seed) @ w
