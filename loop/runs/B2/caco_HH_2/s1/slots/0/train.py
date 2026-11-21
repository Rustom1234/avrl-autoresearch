"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from rdkit.Chem.Scaffolds import MurckoScaffold

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05


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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    r = y - A_tr @ coef
    # Seeded split of training scaffolds into inner-train (80%) and inner-validation (20%)
    groups = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    inner, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X_tr, r, groups))
    params = dict(loss_function='MAE', depth=DEPTH, learning_rate=LR, random_seed=seed, verbose=0, thread_count=8)
    es = CatBoostRegressor(iterations=2000, early_stopping_rounds=100, use_best_model=True, **params)
    es.fit(X_tr[inner], r[inner], eval_set=(X_tr[val], r[val]))
    best = max(1, int(es.get_best_iteration() * 1.1))
    model = CatBoostRegressor(iterations=best, **params)
    model.fit(X_tr, r)
    return model.predict(X_te) + A_te @ coef
