"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
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


ALPHAS = (0.01, 0.1, 1.0)
SHRINK = 5.0


def env_ids(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        ids = set()
        for r in (1, 2):
            ids.update(AllChem.GetMorganFingerprint(m, r).GetNonzeroElements().keys())
        out.append(ids)
    return out


def fit_table(ids_list, resid):
    tot, cnt = {}, {}
    for ids, r in zip(ids_list, resid):
        for k in ids:
            tot[k] = tot.get(k, 0.0) + r
            cnt[k] = cnt.get(k, 0) + 1
    return {k: tot[k] / (cnt[k] + SHRINK) for k in tot}


def encode(ids_list, table):
    out = []
    for ids in ids_list:
        v = [table[k] for k in ids if k in table]
        out.append(np.mean(v) if v else 0.0)
    return np.array(out)


def oof_encode(ids_list, resid, seed):
    enc = np.zeros(len(resid))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(resid):
        table = fit_table([ids_list[i] for i in tr], resid[tr])
        enc[te] = encode([ids_list[i] for i in te], table)
    return enc


def inner_cv_mae(x, resid, alpha, seed):
    err = 0.0
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        pred = Ridge(alpha=alpha).fit(x[tr], resid[tr]).predict(x[te])
        err += np.abs(pred - resid[te]).sum()
    return err


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    ids_tr = env_ids(train_smiles)
    resid = y - A_tr @ coef
    enc_tr = oof_encode(ids_tr, resid, seed)[:, None]
    enc_te = encode(env_ids(test_smiles), fit_table(ids_tr, resid))[:, None]
    alpha = min(ALPHAS, key=lambda a: inner_cv_mae(enc_tr, resid, a, seed))
    ridge = Ridge(alpha=alpha).fit(enc_tr, resid)
    prior_tr = ridge.predict(enc_tr)
    model.fit(featurize(train_smiles), resid - prior_tr)
    return model.predict(featurize(test_smiles)) + A_te @ coef + ridge.predict(enc_te)
