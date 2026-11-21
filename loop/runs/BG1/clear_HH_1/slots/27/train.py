"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
INNER_TREES = 100


def featurize(smiles, kind, radius=RADIUS, nbits=N_BITS):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    if kind == "atompair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
        return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    if kind == "gobbi":
        out = np.zeros((len(mols), N_BITS), dtype=np.float32)
        for i, m in enumerate(mols):
            on = np.array(list(Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory).GetOnBits()), dtype=int)
            out[i, on % N_BITS] = 1
        return out
    return np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])


def rf(seed, n_trees):
    return RandomForestRegressor(n_estimators=n_trees, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_smiles = list(train_smiles)
    test_smiles = list(test_smiles)
    raw = np.asarray(train_y, dtype=np.float64)
    y = rankdata(raw) / len(raw)
    groups = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in train_smiles]
    options = [("morgan", r, b) for r in (1, 2, 3) for b in (1024, 2048)]
    options += [("atompair", 0, N_BITS), ("maccs", 0, 0)]
    scores = []
    for kind, r, b in options:
        X = featurize(train_smiles, kind, r, b)
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=3).split(X, y, groups):
            oof[va] = rf(seed, INNER_TREES).fit(X[tr], y[tr]).predict(X[va])
        scores.append(spearmanr(oof, raw)[0])
    best = np.argsort(scores)[::-1][:2]
    preds = []
    for i in best:
        kind, r, b = options[i]
        model = rf(seed, N_TREES).fit(featurize(train_smiles, kind, r, b), y)
        preds.append(model.predict(featurize(test_smiles, kind, r, b)))
    return np.mean(preds, axis=0)
