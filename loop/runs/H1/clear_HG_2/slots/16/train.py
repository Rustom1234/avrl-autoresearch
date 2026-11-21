"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)])
        feats.append(np.concatenate([fp, desc]))
    return np.array(feats, dtype=np.float32)

def tanimoto_kernel(X):
    # X is (N, D). We apply Tanimoto only to the Morgan bits (first N_BITS)
    X_fp = X[:, :N_BITS]
    dot = np.dot(X_fp, X_fp.T)
    sq_sums = np.sum(X_fp**2, axis=1).reshape(-1, 1)
    return dot / (sq_sums + sq_sums.T - dot + 1e-9)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    # Target transform
    eps = 1e-6
    y_train_log = np.log10(y_train + eps)
    
    # Standardize target
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train_log.reshape(-1, 1)).flatten()
    
    # Simple grid search for alpha
    best_alpha = 1.0
    best_score = float('inf')
    
    # Use first 80% for inner validation
    split = int(len(y_train_scaled) * 0.8)
    X_train_sub = X_train[:split]
    y_train_sub = y_train_scaled[:split]
    X_val_sub = X_train[split:]
    y_val_sub = y_train_scaled[split:]
    
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = KernelRidge(alpha=alpha, kernel='precomputed')
        K = tanimoto_kernel(X_train_sub)
        model.fit(K, y_train_sub)
        
        # Predict on validation (needs kernel between X_train_sub and X_val_sub)
        K_val = tanimoto_kernel(np.hstack([X_train_sub, X_val_sub]))
        preds = model.predict(K_val[:, split:])
        
        # Score (simplified: just mean squared error here as proxy for complexity)
        score = np.mean((preds - y_val_sub)**2)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            
    # Final model
    final_model = KernelRidge(alpha=best_alpha, kernel='precomputed')
    K_final = tanimoto_kernel(X_train)
    final_model.fit(K_final, y_train_scaled)
    
    # Predict test
    X_test = featurize(test_smiles)
    K_test = tanimoto_kernel(np.hstack([X_train, X_test]))
    return final_model.predict(K_test[:, len(train_smiles):])
