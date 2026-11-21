"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05
ALPHAS = [0.03, 0.1, 0.3, 1.0]


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        bv = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)
        arr = np.zeros(N_BITS, dtype=np.float64)
        DataStructs.ConvertToNumpyArray(bv, arr)
        rows.append(arr)
    return np.array(rows)


def scaffold_groups(smiles):
    keys = {}
    groups = []
    for s in smiles:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        groups.append(keys.setdefault(sc, len(keys)))
    return np.array(groups)


def tanimoto(A, B):
    inter = A @ B.T
    denom = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(denom, 1e-12)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    K = tanimoto(X, X)
    groups = scaffold_groups(train_smiles)
    best_alpha, best_err = ALPHAS[0], np.inf
    for alpha in ALPHAS:
        errs = []
        for tr, va in GroupKFold(n_splits=5).split(K, y, groups):
            mu = y[tr].mean()
            mdl = KernelRidge(alpha=alpha, kernel='precomputed')
            mdl.fit(K[np.ix_(tr, tr)], y[tr] - mu)
            pred = mdl.predict(K[np.ix_(va, tr)]) + mu
            errs.append(np.mean(np.abs(pred - y[va])))
        err = np.mean(errs)
        if err < best_err:
            best_alpha, best_err = alpha, err
    mu = y.mean()
    mdl = KernelRidge(alpha=best_alpha, kernel='precomputed')
    mdl.fit(K, y - mu)
    return mdl.predict(tanimoto(Xt, X)) + mu
