"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb

# Settings
N_BITS = 1024
RADIUS = 2


def _clean(rows):
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def featurize(smiles):
    """Returns three feature blocks: descriptors, Morgan counts, atom-pair counts."""
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=False)
    ap = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    d, f, a = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d.append([fn(m) for _, fn in Descriptors.descList])
        f.append(mg.GetCountFingerprintAsNumPy(m))
        a.append(ap.GetCountFingerprintAsNumPy(m))
    return [_clean(d), _clean(f), _clean(a)]


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    preds = []
    for a, b in zip(Xtr, Xte):
        model = lgb.LGBMRegressor(objective="huber", alpha=1.0, learning_rate=0.03, n_estimators=1000,
                                  colsample_bytree=0.3, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                  n_jobs=8, random_state=seed, verbose=-1)
        model.fit(a, y)
        preds.append(model.predict(b))
    return np.mean(preds, axis=0)
