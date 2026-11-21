"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def tanimoto_kernel(X1, X2):
    dot = np.dot(X1, X2.T)
    sum_sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sum_sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
    return dot / (sum_sq1 + sum_sq2 - dot + 1e-9)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    y_train = np.asarray(train_y, dtype=np.float64)
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)

    # Forest component
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    
    # kNN component
    K_test = tanimoto_kernel(X_test, X_train)
    
    if np.all(y_train >= 0):
        y_train_trans = np.log1p(y_train)
        
        rf.fit(X_train, y_train_trans)
        rf_preds = rf.predict(X_test)
        
        # Weighted kNN
        k = 7
        # Get top k indices and similarities
        top_k_indices = np.argsort(K_test, axis=1)[:, -k:]
        top_k_sims = np.take_along_axis(K_test, top_k_indices, axis=1)
        
        # Weights = sim^3
        weights = np.power(top_k_sims, 3)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        
        # Average of top k train targets
        y_train_trans_2d = y_train_trans.reshape(-1, 1)
        k_train_targets = np.take_along_axis(y_train_trans_2d, top_k_indices, axis=1)
        knn_preds = np.sum(k_train_targets * weights, axis=1) / (weight_sums + 1e-9)
        
        # Blend
        blended = 0.6 * rf_preds + 0.4 * knn_preds
        return np.expm1(blended)
    else:
        rf.fit(X_train, y_train)
        return rf.predict(X_test)
