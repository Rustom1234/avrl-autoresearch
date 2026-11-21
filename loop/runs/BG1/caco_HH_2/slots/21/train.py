"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_groups(smiles):
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in smiles]
    return np.unique(scaf, return_inverse=True)[1]


def bit_fps(smiles):
    return [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), RADIUS, nBits=N_BITS) for s in smiles]

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
    X_tr = featurize(train_smiles)
    X_te = featurize(test_smiles)
    target = y - A_tr @ coef

    def make():
        return CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                                 random_seed=seed, verbose=0, thread_count=8)

    model = make()
    model.fit(X_tr, target)
    pred = model.predict(X_te) + A_te @ coef

    # inner 4-fold out-of-fold residuals of the boosting model, grouped by Murcko scaffold
    oof = np.zeros(len(y))
    for tr_idx, va_idx in GroupKFold(n_splits=4).split(X_tr, target, scaffold_groups(train_smiles)):
        inner = make()
        inner.fit(X_tr[tr_idx], target[tr_idx])
        oof[va_idx] = inner.predict(X_tr[va_idx])
    resid = target - oof

    # Tanimoto-neighbour residual correction: shrink 0.5, skip when max similarity < 0.3
    tr_fp = bit_fps(train_smiles)
    for i, fp in enumerate(bit_fps(test_smiles)):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, tr_fp))
        nn = np.argsort(-sims)[:5]
        if sims[nn[0]] < 0.3:
            continue
        w = sims[nn]
        pred[i] += 0.5 * np.sum(w * resid[nn]) / np.sum(w)
    return pred
