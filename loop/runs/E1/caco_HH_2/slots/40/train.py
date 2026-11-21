"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor
from sklearn.linear_model import QuantileRegressor
from rdkit.Chem import rdMolDescriptors

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


def crippen_counts(smiles, vocab=None):
    keys = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        contribs = rdMolDescriptors._CalcCrippenContribs(m)
        keys.append([(round(a, 3), round(b, 3)) for a, b in contribs])
    if vocab is None:
        vocab = {}
        for ks in keys:
            for k in ks:
                vocab.setdefault(k, len(vocab))
    X = np.zeros((len(keys), len(vocab)))
    for i, ks in enumerate(keys):
        for k in ks:
            j = vocab.get(k)
            if j is not None:
                X[i, j] += 1
    return X, vocab


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    C_tr, vocab = crippen_counts(train_smiles)
    C_te, _ = crippen_counts(test_smiles, vocab)
    A_tr = np.c_[C_tr, base_features(train_smiles)[:, :2] / 100.0]
    A_te = np.c_[C_te, base_features(test_smiles)[:, :2] / 100.0]
    prior = QuantileRegressor(quantile=0.5, alpha=1e-3, solver='highs').fit(A_tr, y)
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - prior.predict(A_tr))
    return model.predict(featurize(test_smiles)) + prior.predict(A_te)
