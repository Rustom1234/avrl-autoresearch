"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

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


N_SUB = 50
KEEP_FRAC = 0.6


def rdkit_desc(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([f(m) for _, f in Descriptors.descList])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    n = len(y)
    X_tr = np.c_[base_features(train_smiles), rdkit_desc(train_smiles)]
    X_te = np.c_[base_features(test_smiles), rdkit_desc(test_smiles)]
    scaler = StandardScaler().fit(X_tr)
    Z_tr = np.clip(scaler.transform(X_tr), -10, 10)
    Z_te = np.clip(scaler.transform(X_te), -10, 10)
    rng = np.random.RandomState(seed)
    counts = np.zeros(Z_tr.shape[1])
    for _ in range(N_SUB):
        idx = rng.choice(n, n // 2, replace=False)
        lasso = LassoCV(cv=3, n_alphas=30, max_iter=5000, random_state=seed).fit(Z_tr[idx], y[idx])
        counts += lasso.coef_ != 0
    keep = np.where(counts / N_SUB >= KEEP_FRAC)[0]
    if len(keep) == 0:
        keep = np.arange(Z_tr.shape[1])
    A_tr = np.c_[np.ones(n), Z_tr[:, keep]]
    A_te = np.c_[np.ones(len(test_smiles)), Z_te[:, keep]]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - A_tr @ coef)
    return model.predict(featurize(test_smiles)) + A_te @ coef
