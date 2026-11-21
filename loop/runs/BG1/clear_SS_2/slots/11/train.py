"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        desc = np.array([f(m) for _, f in Descriptors._descList], dtype=np.float64)
        ffp = fgen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        rows.append(np.concatenate([desc, fp, ffp]))
    X = np.array(rows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    nd = len(Descriptors._descList)
    model.fit(Xtr[:, nd:], y)
    p_rf = model.predict(Xte[:, nd:])
    gbm = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.5, min_child_samples=10, n_jobs=8, random_state=seed, verbose=-1)
    gbm.fit(Xtr[:, :nd], y)
    p_gbm = gbm.predict(Xte[:, :nd])
    return 0.5 * rankdata(p_rf) + 0.5 * rankdata(p_gbm)
