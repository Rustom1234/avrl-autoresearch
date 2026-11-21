"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Settings
N_BITS = 2048
N_TREES_RF = 300
N_TREES_XGB = 300
N_TREES_LGBM = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)
            arr = np.zeros((1, N_BITS), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr[0])
        else:
            fps.append(np.zeros((N_BITS,), dtype=np.float32))
    return np.array(fps)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)

    rf = RandomForestRegressor(n_estimators=N_TREES_RF, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    
    xgb = XGBRegressor(n_estimators=N_TREES_XGB, max_depth=6, learning_rate=0.1, random_state=seed)
    xgb.fit(X_train, y_train)
    
    lgbm = LGBMRegressor(n_estimators=N_TREES_LGBM, max_depth=6, learning_rate=0.1, random_state=seed)
    lgbm.fit(X_train, y_train)
    
    return (rf.predict(X_test) + xgb.predict(X_test) + lgbm.predict(X_test)) / 3
