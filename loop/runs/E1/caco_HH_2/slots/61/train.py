"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import QuantileRegressor

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
    B_tr = base_features(train_smiles)
    B_te = base_features(test_smiles)
    # base_features columns: 0 TPSA, 1 MolWt, 2 NumHDonors, 3 MolLogP
    knots = {c: np.percentile(B_tr[:, c], [25, 50, 75]) for c in (0, 3)}

    def design(B):
        cols = [B[:, 1], B[:, 2]]
        for c in (0, 3):
            cols.append(B[:, c])
            cols += [np.maximum(B[:, c] - k, 0) for k in knots[c]]
        return np.column_stack(cols)

    A_tr = design(B_tr)
    A_te = design(B_te)
    prior_model = QuantileRegressor(quantile=0.5, alpha=0.0)
    prior_model.fit(A_tr, y)
    prior_tr = prior_model.predict(A_tr)
    prior_te = prior_model.predict(A_te)
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - prior_tr)
    return model.predict(featurize(test_smiles)) + prior_te
