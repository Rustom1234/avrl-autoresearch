"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import xgboost as xgb

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
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'min_child_weight': 3,
        'tree_method': 'hist',
        'random_state': seed,
        'nthread': 8
    }

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
        model = xgb.XGBRegressor(**params, n_estimators=500)
        model.fit(X_train, y_train_transformed)
        preds = model.predict(X_test)
        return np.expm1(preds)
    else:
        model = xgb.XGBRegressor(**params, n_estimators=500)
        model.fit(X_train, y_train)
        return model.predict(X_test)
