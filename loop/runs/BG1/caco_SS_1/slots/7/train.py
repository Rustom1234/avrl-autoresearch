"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        cnt = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, cnt]))
    X = np.nan_to_num(np.array(rows), nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    nd = len(Descriptors.descList)
    svd = TruncatedSVD(n_components=50, random_state=seed)
    svd.fit(np.vstack([Xtr[:, nd:], Xte[:, nd:]]))
    Xtr = np.hstack([Xtr[:, :nd], svd.transform(Xtr[:, nd:])])
    Xte = np.hstack([Xte[:, :nd], svd.transform(Xte[:, nd:])])
    model = LGBMRegressor(n_estimators=600, num_leaves=15, learning_rate=0.03, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.4, reg_lambda=5, min_child_samples=10,
                          n_jobs=8, random_state=seed, verbose=-1)
    model.fit(Xtr, np.asarray(train_y, dtype=np.float64))
    return model.predict(Xte)
