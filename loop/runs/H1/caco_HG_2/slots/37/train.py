"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import QuantileRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 700
LEARNING_RATE = 0.03

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
    
    # Median regression on descriptors
    scaler = StandardScaler()
    X_desc_scaled = scaler.fit_transform(X_desc)
    
    # Out-of-fold residuals for XGB
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof_res = np.zeros_like(y_all)
    
    # Median model to predict residuals
    base = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
    for tr_idx, val_idx in kf.split(X_desc_scaled):
        X_train_r = X_desc_scaled[tr_idx]
        X_val_r = X_desc_scaled[val_idx]
        y_train_r = y_all[tr_idx]
        
        base.fit(X_train_r, y_train_r)
        oof_res[val_idx] = y_all[val_idx] - base.predict(X_val_r)
    
    # XGB on fingerprints + descriptors to predict residuals
    xgb_model = XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, 
                               max_depth=6, subsample=0.8, colsample_bytree=0.5, 
                               objective='reg:absoluteerror', n_jobs=-1, random_state=seed)
    xgb_model.fit(X_all, oof_res)
    
    # Final Prediction
    X_test = featurize(test_smiles)
    X_test_desc = X_test[:, -5:]
    base_pred = base.predict(scaler.transform(X_test_desc))
    xgb_res_pred = xgb_model.predict(X_test)
    
    return base_pred + xgb_res_pred
