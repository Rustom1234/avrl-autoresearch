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
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m)])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


K_NN = 100


def local_prior(Z_q, Z_ref, y_ref, self_ref):
    D = ((Z_q[:, None, :] - Z_ref[None, :, :]) ** 2).sum(-1)
    if self_ref:
        np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :K_NN]
    out = np.empty(len(Z_q))
    for i in range(len(Z_q)):
        j = idx[i]
        d = np.sqrt(D[i, j])
        h = d[-1] + 1e-9
        sw = np.exp(-0.25 * (d / h) ** 2)  # square root of Gaussian kernel weight
        A = np.c_[np.ones(K_NN), Z_ref[j]] * sw[:, None]
        coef = np.linalg.lstsq(A, y_ref[j] * sw, rcond=None)[0]
        out[i] = coef[0] + Z_q[i] @ coef[1:]
    return out


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    B_tr = base_features(train_smiles)
    B_te = base_features(test_smiles)
    mu = B_tr.mean(0)
    sd = B_tr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z_tr = (B_tr - mu) / sd
    Z_te = (B_te - mu) / sd
    prior_tr = local_prior(Z_tr, Z_tr, y, True)
    prior_te = local_prior(Z_te, Z_tr, y, False)
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - prior_tr)
    return model.predict(featurize(test_smiles)) + prior_te
