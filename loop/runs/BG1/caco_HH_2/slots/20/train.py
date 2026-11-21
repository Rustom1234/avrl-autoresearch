"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05
INNER_ITERS = 400


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
    r = y - A_tr @ coef
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)

    def stage1(n_iter):
        return CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=n_iter,
                                 random_seed=seed, verbose=0, thread_count=8)

    # Stage 1: inner 4-fold out-of-fold predictions of the OLS residual
    oof = np.zeros(len(y))
    for tr_idx, va_idx in KFold(4, shuffle=True, random_state=seed).split(X_tr):
        m = stage1(INNER_ITERS)
        m.fit(X_tr[tr_idx], r[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
    full = stage1(ITERS)
    full.fit(X_tr, r)
    p1 = full.predict(X_te)

    # Stage 2: shallower model on the out-of-fold residuals
    stage2 = CatBoostRegressor(loss_function='MAE', depth=4, learning_rate=0.03, iterations=300,
                               random_seed=seed, verbose=0, thread_count=8)
    stage2.fit(X_tr, r - oof)
    return p1 + stage2.predict(X_te) + A_te @ coef
