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


ACID_PATS = [Chem.MolFromSmarts(s) for s in ['[CX3](=O)[OX2H1,OX1-]', '[#6]1:[#7]:[#7]:[#7]:[#7]:1', 'C(=O)[NH]S(=O)=O']]
BASE_PATS = [Chem.MolFromSmarts('[NX3;!$(N~[!#6;!#1]);!$(N-C=[O,S,N]);!$(N-a);!$(N-C=C)]')]
QUAT_PATS = [Chem.MolFromSmarts('[N+;!$([N+]-[O-]);!$([N+]=O)]')]
PHENOL_PATS = [Chem.MolFromSmarts('c[OX2H1]')]


def ion_counts(m):
    return [sum(len(m.GetSubstructMatches(p)) for p in pats) for pats in (ACID_PATS, BASE_PATS, QUAT_PATS, PHENOL_PATS)]


def base_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m)]
                    + ion_counts(m))
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - A_tr @ coef)
    return model.predict(featurize(test_smiles)) + A_te @ coef
