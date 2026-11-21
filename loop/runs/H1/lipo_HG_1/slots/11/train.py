"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
C = 10.0
EPS = 0.1

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    # We need fingerprints as bit vectors for Tanimoto, but GetFingerprintAsNumPy returns floats.
    # However, SVR with precomputed kernel just needs a similarity matrix.
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    def get_tanimoto(A, B):
        dot_prod = np.dot(A, B.T)
        sum_A = np.sum(A, axis=1).reshape(-1, 1)
        sum_B = np.sum(B, axis=1).reshape(1, -1)
        return dot_prod / (sum_A + sum_B - dot_prod + 1e-9)

    # Inner OOF split for recalibration
    n_train = len(train_smiles)
    indices = np.random.permutation(n_train)
    rng = np.random.RandomState(seed)
    # Shuffle indices using the seed
    indices = rng.permutation(n_train)
    
    # Simple 3-fold OOF
    folds = np.array_split(indices, 3)
    oof_preds = np.zeros(n_train)
    
    # Use only first 2 folds for OOF to keep it simple and fast
    for i in range(2):
        train_idx = np.concatenate([folds[j] for j in range(3) if j != i])
        val_idx = folds[i]
        
        K_train_inner = get_tanimoto(X_train[train_idx], X_train[train_idx])
        K_val_inner = get_tanimoto(X_train[val_idx], X_train[train_idx])
        
        model_inner = SVR(kernel='precomputed', C=C, epsilon=EPS)
        model_inner.fit(K_train_inner, y_train[train_idx])
        oof_preds[val_idx] = model_inner.predict(K_val_inner)
    
    # Fit linear recalibration: y = a * oof_pred + b
    # Use simple OLS for speed/stability
    A = np.stack([oof_preds, np.ones(n_train)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)
    a, b = sol[0], sol[1]
    
    # Final model on all data
    K_train_full = get_tanimoto(X_train, X_train)
    K_test_full = get_tanimoto(X_test, X_train)
    
    model_full = SVR(kernel='precomputed', C=C, epsilon=EPS)
    model_full.fit(K_train_full, y_train)
    
    raw_preds = model_full.predict(K_test_full)
    return a * raw_preds + b
