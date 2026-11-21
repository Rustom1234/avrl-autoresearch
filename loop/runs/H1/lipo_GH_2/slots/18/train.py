"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


from rdkit.Chem import Descriptors

_DESC_CACHE = {}


def featurize(smiles):
    for s in smiles:
        if s not in _DESC_CACHE:
            mol = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(mol) for _, f in Descriptors.descList]
    X = np.array([_DESC_CACHE[s] for s in smiles], dtype=np.float64)
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)
    return np.hstack([X, fps]).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    X_test = featurize(test_smiles)
    preds = []
    for k in range(3):
        model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5,
                             tree_method="hist", n_jobs=8, random_state=seed + k)
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    return np.mean(preds, axis=0)
