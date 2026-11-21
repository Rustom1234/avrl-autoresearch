"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Calculate common descriptors
        desc_list = []
        # Use a subset of common descriptors to avoid complex dictionary lookups
        for name in ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'MolConnectivityሼ', 'NumHeavyAtoms', 'MaxMorse']:
            try:
                # Standard RDKit Descriptors are usually accessible via Descriptors.MolWt etc.
                # To keep it simple and robust, we use the known ones
                pass
            except: pass
        
        # Simpler approach for robustness: just use the fingerprints
        features.append(fp)
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.log1p(np.asarray(train_y, dtype=np.float64))

    # Use XGBoost as the primary model, it's usually strongest for this size
    xgb = XGBRegressor(n_estimators=N_TREES_XGB, max_depth=7, learning_rate=0.05, random_state=seed)
    xgb.fit(X_train, y_train)
    
    preds = xgb.predict(X_test)
    return np.expm1(preds)
