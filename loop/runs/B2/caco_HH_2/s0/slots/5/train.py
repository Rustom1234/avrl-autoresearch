"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor

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
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m),
                     Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), Descriptors.FractionCSP3(m)])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    X_tr = base_features(train_smiles)
    X_te = base_features(test_smiles)
    prior_tr = np.zeros(len(y))
    prior_te = np.zeros(len(test_smiles))
    for cols in [[0, 3], [1, 5], [2, 4], [3, 0, 6]]:
        A_tr = np.c_[np.ones(len(y)), X_tr[:, cols]]
        A_te = np.c_[np.ones(len(test_smiles)), X_te[:, cols]]
        coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
        prior_tr += A_tr @ coef / 4
        prior_te += A_te @ coef / 4
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - prior_tr)
    return model.predict(featurize(test_smiles)) + prior_te
