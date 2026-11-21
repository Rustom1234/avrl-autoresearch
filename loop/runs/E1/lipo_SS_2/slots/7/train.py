"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e5, 1e5).astype(np.float32)
        rows.append(np.concatenate([fp, desc]))
    return np.array(rows, dtype=np.float32)


def crippen(smiles):
    return np.array([Descriptors.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64) - crippen(train_smiles)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.1, random_state=seed)
    model = lgb.LGBMRegressor(objective="l1", learning_rate=0.01, n_estimators=3000, num_leaves=15,
                              min_child_samples=10, reg_lambda=5, colsample_bytree=0.3,
                              n_jobs=8, random_state=seed, verbose=-1)
    model.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model.predict(featurize(test_smiles)) + crippen(test_smiles)
