"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

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


def morgan_counts(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    Ftr, Fte = morgan_counts(train_smiles), morgan_counts(test_smiles)
    rf_d = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                 random_state=seed, n_jobs=8).fit(Xtr, y)
    cb = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, loss_function="MAE", random_seed=seed,
                           thread_count=8, verbose=0).fit(Xtr, y)
    rf_f = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                 random_state=seed, n_jobs=8).fit(Ftr, y)
    return (rf_d.predict(Xte) + cb.predict(Xte) + rf_f.predict(Fte)) / 3.0
