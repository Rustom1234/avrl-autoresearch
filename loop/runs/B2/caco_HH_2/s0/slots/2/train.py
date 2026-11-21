"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05
ALPHAS = [1, 10, 100, 1000]


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = np.zeros(N_BITS, dtype=np.float32)
        for k, v in AllChem.GetHashedMorganFingerprint(m, RADIUS, nBits=N_BITS).GetNonzeroElements().items():
            fp[k] = v
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float32)
        rows.append(np.concatenate([fp, desc]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def base_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m)])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def desc_matrix(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def scaffold_groups(smiles):
    keys = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles]
    codes = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([codes[k] for k in keys])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    Dtr = desc_matrix(train_smiles)
    Dte = desc_matrix(test_smiles)
    keep = Dtr.std(axis=0) > 0
    Dtr, Dte = Dtr[:, keep], Dte[:, keep]
    mu, sd = Dtr.mean(axis=0), Dtr.std(axis=0)
    Ztr = (Dtr - mu) / sd
    Zte = (Dte - mu) / sd
    groups = scaffold_groups(train_smiles)
    best_a, best_err = ALPHAS[0], np.inf
    for a in ALPHAS:
        err = 0.0
        for tr, va in GroupKFold(n_splits=3).split(Ztr, y, groups):
            r = Ridge(alpha=a).fit(Ztr[tr], y[tr])
            err += np.abs(r.predict(Ztr[va]) - y[va]).sum()
        if err < best_err:
            best_a, best_err = a, err
    ridge = Ridge(alpha=best_a).fit(Ztr, y)
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - ridge.predict(Ztr))
    return model.predict(featurize(test_smiles)) + ridge.predict(Zte)
