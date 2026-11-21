"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = CatBoostRegressor(
        iterations=800, 
        depth=6, 
        learning_rate=0.05, 
        l2_leaf_reg=5, 
        rsm=0.3, 
        random_seed=seed, 
        verbose=0, 
        thread_count=8
    )
    y_train = np.asarray(train_y, dtype=np.float64)
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
        model.fit(X_train, y_train_transformed)
        preds = model.predict(X_test)
        return np.expm1(preds)
    else:
        model.fit(X_train, y_train)
        return model.predict(X_test)
