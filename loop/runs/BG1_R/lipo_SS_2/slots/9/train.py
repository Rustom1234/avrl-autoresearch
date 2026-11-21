"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
    model = ExtraTreesRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64) - crippen(train_smiles)
    lgb = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, colsample_bytree=0.3,
                        subsample=0.8, subsample_freq=1, min_child_samples=10, objective="l1",
                        n_jobs=8, random_state=seed, verbose=-1)
    cat = CatBoostRegressor(iterations=500, learning_rate=0.06, depth=6, loss_function="MAE",
                            rsm=0.2, border_count=32, thread_count=8, random_seed=seed, verbose=False)
    preds = []
    for m in (model, lgb, cat):
        m.fit(Xtr, y)
        preds.append(np.asarray(m.predict(Xte), dtype=np.float64))
    return np.mean(preds, axis=0) + crippen(test_smiles)
