"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y_train = np.asarray(train_y, dtype=np.float64)
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'min_child_samples': 10,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.3,
        'random_state': seed,
        'n_jobs': 8,
        'verbosity': -1
    }

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
        train_data = lgb.Dataset(X_train, label=y_train_transformed)
        model = lgb.train({'params': params, 'num_boost_round': 600}, train_data)
        preds = model.predict(X_test)
        return np.expm1(preds)
    else:
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train({'params': params, 'num_boost_round': 600}, train_data)
        return model.predict(X_test)
