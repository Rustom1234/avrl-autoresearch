"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Append NumAtoms and TPSA
        features.append(np.append(fp, [mol.GetNumAtoms(), Descriptors.TPSA(mol)]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.log1p(np.asarray(train_y, dtype=np.float64))
    
    m1 = XGBRegressor(n_estimators=N_TREES, max_depth=6, 
                        learning_rate=0.1, subsample=0.8, 
                        colsample_bytree=MAX_FEATURES, 
                        n_jobs=-1, random_state=seed)
    m1.fit(X_train, y_train)
    
    m2 = RandomForestRegressor(n_estimators=N_TREES, max_depth=10, 
                                 min_samples_split=5, 
                                 n_jobs=-1, random_state=seed)
    m2.fit(X_train, y_train)
    
    m3 = CatBoostRegressor(iterations=N_TREES, depth=6, 
                            learning_rate=0.1, verbose=0, 
                            random_seed=seed)
    m3.fit(X_train, y_train)
    
    p1 = m1.predict(X_test)
    p2 = m2.predict(X_test)
    p3 = m3.predict(X_test)
    
    return np.expm1((p1 + p2 + p3) / 3)
