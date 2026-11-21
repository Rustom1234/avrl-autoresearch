"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
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


def make_model(seed):
    return CatBoostRegressor(loss_function='RMSE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                             random_seed=seed, verbose=0, thread_count=8)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    r = y - A_tr @ coef
    # median bias correction from scaffold-grouped out-of-fold residual predictions
    scaff = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    groups = np.unique(scaff, return_inverse=True)[1]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=4).split(X_tr, r, groups):
        oof[te] = make_model(seed).fit(X_tr[tr], r[tr]).predict(X_tr[te])
    m = np.median(r - oof)
    model = make_model(seed).fit(X_tr, r)
    return model.predict(X_te) + A_te @ coef + m
