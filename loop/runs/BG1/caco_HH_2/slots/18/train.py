"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    X_tr = featurize(train_smiles).astype(np.float64)
    X_te = featurize(test_smiles).astype(np.float64)
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(X_tr, y - A_tr @ coef)
    tree_pred = model.predict(X_te) + A_te @ coef
    # Linear ElasticNet on counts + descriptors, constant columns removed
    keep = X_tr.std(axis=0) > 0
    scaler = StandardScaler().fit(X_tr[:, keep])
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=KFold(5, shuffle=True, random_state=seed),
                        max_iter=5000, n_jobs=8)
    enet.fit(scaler.transform(X_tr[:, keep]), y)
    lin_pred = enet.predict(scaler.transform(X_te[:, keep]))
    return 0.5 * (tree_pred + lin_pred)
