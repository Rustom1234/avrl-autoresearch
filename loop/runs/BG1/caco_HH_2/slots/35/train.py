"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge

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


def desc_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    D_tr = desc_features(train_smiles)
    D_te = desc_features(test_smiles)
    keep = D_tr.std(axis=0) > 0
    mu = D_tr[:, keep].mean(axis=0)
    sd = D_tr[:, keep].std(axis=0)
    Z_tr = (D_tr[:, keep] - mu) / sd
    Z_te = (D_te[:, keep] - mu) / sd
    nys = Nystroem(kernel='rbf', gamma=1.0 / Z_tr.shape[1], n_components=300, random_state=seed)
    Phi_tr = nys.fit_transform(Z_tr)
    Phi_te = nys.transform(Z_te)
    ridge = Ridge(alpha=10.0).fit(Phi_tr, y)
    r_tr = ridge.predict(Phi_tr)
    r_te = ridge.predict(Phi_te)
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - r_tr)
    return model.predict(featurize(test_smiles)) + r_te
