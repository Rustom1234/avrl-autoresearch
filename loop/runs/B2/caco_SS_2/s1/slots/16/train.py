"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


ACID_PATS = [Chem.MolFromSmarts(s) for s in [
    "[CX3](=O)[OX2H1,OX1-]", "c1nn[nH]n1", "c1n[nH]nn1", "C(=O)[NH]S(=O)(=O)",
    "P(=O)[OH]", "S(=O)(=O)[OH]"]]
BASE_PATS = [Chem.MolFromSmarts(s) for s in [
    "[NX3;!$(N-[!CX4])]", "[CX3](=[NX2;!a;!$(N-[O,S])])[NX3;!a;!$(NC=O)]", "[NX4+]"]]


def ion_features(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        na = sum(len(m.GetSubstructMatches(p)) for p in ACID_PATS)
        nb = sum(len(m.GetSubstructMatches(p)) for p in BASE_PATS)
        tpsa = Descriptors.TPSA(m)
        ha = m.GetNumHeavyAtoms()
        hbd = Descriptors.NumHDonors(m)
        logp = Descriptors.MolLogP(m)
        out.append([na, nb, nb - na, float(na > 0 and nb > 0), float(na == 0 and nb == 0),
                    tpsa, Descriptors.MolWt(m), hbd / max(ha, 1), logp - tpsa / 50.0])
    return np.array(out, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    Xtr = np.hstack([featurize(train_smiles), ion_features(train_smiles)])
    Xte = np.hstack([featurize(test_smiles), ion_features(test_smiles)])
    y = np.asarray(train_y, dtype=np.float64)
    model.fit(Xtr, y)
    p_lgb = model.predict(Xte)
    keep = Xtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
    Str = qt.fit_transform(Xtr[:, keep])
    Ste = qt.transform(Xte[:, keep])
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Ste)
