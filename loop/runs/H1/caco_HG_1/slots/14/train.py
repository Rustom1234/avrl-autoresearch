"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostRegressor
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.optimize import minimize

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Filter out None
    mols = [m for m in mols if m is not None]
    smiles = [s for s, m in zip(smiles, mols) if m is not None]
    
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    # Basic Physicochemical Descriptors
    desc_list = []
    for m in mols:
        d = [
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            Descriptors.NumHDonors(m),
            Descriptors.NumHAcceptors(m),
            Descriptors.HeavyAtomCount(m),
            Descriptors.TPSA(m)
        ]
        desc_list.append(d)
    
    desc_array = np.array(desc_list, dtype=np.float32)
    # Handle NaNs by replacing with 0
    desc_array = np.nan_to_num(desc_array)
    
    # Concatenate Morgan + Descriptors
    return np.hstack([fps, desc_array])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    
    # Define base models
    def get_cb_model():
        return CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, 
                                  loss_function='MAE', random_seed=seed, verbose=0)
    
    def get_rf_model():
        return RandomForestRegressor(n_estimators=100, max_depth=12, 
                                      random_state=seed, n_jobs=-1)
    
    # Inner 3-fold CV to find blend weights
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    oof_cb = np.zeros(len(y))
    oof_rf = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        m_cb = get_cb_model()
        m_cb.fit(X_tr, y_tr)
        oof_cb[val_idx] = m_cb.predict(X_val)
        
        m_rf = get_rf_model()
        m_rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = m_rf.predict(X_val)
    
    # Optimize weights
    def objective(w):
        preds = w[0] * oof_cb + w[1] * oof_rf
        return np.mean(np.abs(preds - y))
    
    res = minimize(objective, x0=[0.5, 0.5], 
                    bounds=[(0, 1), (0, 1)], 
                    method='SLSQP', constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    weights = res.x
    
    # Final training
    m_cb_final = get_cb_model()
    m_cb_final.fit(X_train, y)
    
    m_rf_final = get_rf_model()
    m_rf_final.fit(X_train, y)
    
    # Weighted prediction
    cb_preds = m_cb_final.predict(X_test)
    rf_preds = m_rf_final.predict(X_test)
    
    return weights[0] * cb_preds + weights[1] * rf_preds
