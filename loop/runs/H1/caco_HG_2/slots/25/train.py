"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 2000
LEARNING_RATE = 0.01

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, 
        fpSize=N_BITS, 
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Morgan fingerprints
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    # RDKit descriptors
    desc = np.array([[Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m), 
                               Descriptors.NumHAcceptors(m), Descriptors.TPSA(m)] 
                      for m in mols], dtype=np.float32)
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_all = featurize(train_smiles)
    y_all = np.asarray(train_y, dtype=np.float64)
    
    # Split features for Ridge
    X_desc = X_all[:, -5:]
    
    # Ridge on descriptors
    scaler = StandardScaler()
    X_desc_scaled = scaler.fit_transform(X_desc)
    
    # Out-of-fold residuals for LGBM
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_res = np.zeros_like(y_all)
    
    # Ridge model to predict residuals
    ridge = Ridge(alpha=1.0)
    for tr_idx, val_idx in kf.split(X_desc_scaled):
        X_train_r = X_desc_scaled[tr_idx]
        X_val_r = X_desc_scaled[val_idx]
        y_train_r = y_all[tr_idx]
        
        ridge.fit(X_train_r, y_train_r)
        oof_res[val_idx] = y_all[val_idx] - ridge.predict(X_val_r)
    
    # LGBM on fingerprints + descriptors to predict residuals
    lgbm_model = LGBMRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, 
                                 max_depth=6, subsample=0.8, 
                                 objective='regression_l1', n_jobs=-1, random_state=seed)
    lgbm_model.fit(X_all, oof_res)
    
    # Final Prediction
    X_test = featurize(test_smiles)
    X_test_desc = X_test[:, -5:]
    ridge_pred = ridge.predict(scaler.transform(X_test_desc))
    lgbm_res_pred = lgbm_model.predict(X_test)
    
    return ridge_pred + lgbm_res_pred
