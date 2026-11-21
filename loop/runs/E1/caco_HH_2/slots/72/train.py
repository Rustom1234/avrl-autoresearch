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
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float32)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    blocks = []
    for radius, chiral in [(RADIUS, False), (RADIUS, True), (3, False)]:
        fp = np.zeros((len(mols), N_BITS), dtype=np.float32)
        for i, m in enumerate(mols):
            for k, v in AllChem.GetHashedMorganFingerprint(m, radius, nBits=N_BITS, useChirality=chiral).GetNonzeroElements().items():
                fp[i, k] = v
        blocks.append(np.concatenate([fp, desc], axis=1))
    return blocks


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
    resid = y - A_tr @ coef
    preds = []
    for Xtr, Xte in zip(featurize(train_smiles), featurize(test_smiles)):
        model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                                  random_seed=seed, verbose=0, thread_count=8)
        model.fit(Xtr, resid)
        preds.append(model.predict(Xte))
    return np.mean(preds, axis=0) + A_te @ coef
