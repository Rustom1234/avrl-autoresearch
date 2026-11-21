"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, fpSize=N_BITS,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=seed)
    params = dict(objective="regression", learning_rate=0.01, num_leaves=31, min_child_samples=5,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, seed=seed, verbose=-1, num_threads=8)
    booster = lgb.train(params, lgb.Dataset(X_tr, y_tr), num_boost_round=2000,
                        valid_sets=[lgb.Dataset(X_val, y_val)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    n_rounds = max(booster.best_iteration, 1)
    model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_rounds)
    return model.predict(featurize(test_smiles))
