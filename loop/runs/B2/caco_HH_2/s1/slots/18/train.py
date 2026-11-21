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
    X = featurize(train_smiles)
    X_te = featurize(test_smiles)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    target = y - A_tr @ coef
    # 5-fold out-of-fold predictions of the residual target, used to find the rows to drop
    oof = np.zeros(len(y))
    for tr_idx, va_idx in KFold(5, shuffle=True, random_state=seed).split(X):
        cv_model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=300,
                                     random_seed=seed, verbose=0, thread_count=8)
        cv_model.fit(X[tr_idx], target[tr_idx])
        oof[va_idx] = cv_model.predict(X[va_idx])
    # keep the 95% of rows with the smallest absolute OOF residual
    keep = np.argsort(np.abs(target - oof))[:int(0.95 * len(y))]
    coef = np.linalg.lstsq(A_tr[keep], y[keep], rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(X[keep], y[keep] - A_tr[keep] @ coef)
    return model.predict(X_te) + A_te @ coef
