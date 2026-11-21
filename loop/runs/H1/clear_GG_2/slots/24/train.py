"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import rankdata

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES_RF = 300
N_TREES_XGB = 300
N_TREES_LGBM = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    maccs_gen = MACCSkeys.GenMACCSKeys
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        mf = gen.GetFingerprintAsNumPy(mol)
        mk = np.array(maccs_gen(mol), dtype=np.float32)
        ap = ap_gen.GetFingerprintAsNumPy(mol)
        features.append(np.concatenate([mf, mk, ap]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train_raw = np.asarray(train_y, dtype=np.float64)
    y_train = rankdata(y_train_raw)

    rf = RandomForestRegressor(n_estimators=N_TREES_RF, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    
    xgb = XGBRegressor(n_estimators=N_TREES_XGB, max_depth=6, learning_rate=0.1, random_state=seed)
    xgb.fit(X_train, y_train)
    
    lgbm = LGBMRegressor(n_estimators=N_TREES_LGBM, max_depth=6, learning_rate=0.1, random_state=seed)
    lgbm.fit(X_train, y_train)
    
    preds = (rf.predict(X_test) + xgb.predict(X_test) + lgbm.predict(X_test)) / 3
    
    # Map rank-space predictions back to original space
    sorted_indices = np.argsort(y_train_raw)
    sorted_preds = preds[sorted_indices]
    # This is a simplified rank-to-value mapping; for higher precision we use linear interpolation
    # but since we only need a continuous mapping back, we'll use the sorted original values
    return np.interp(preds, y_train, y_train_raw)
