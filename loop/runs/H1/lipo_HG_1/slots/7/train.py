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
    
    # Compute Tanimoto kernel: K(x,y) = sum(min(x,y)) / sum(max(x,y))
    # For binary fingerprints, this is the standard Jaccard similarity.
    # We use a simple efficient approximation: 
    # Since we have float vectors, we'll use a manual implementation of the kernel.
    # To stay within memory limits, we avoid huge intermediates.
    
    def get_tanimoto(A, B):
        # A: (n, d), B: (m, d)
        # Tanimoto = dot(A, B) / (sum(A) + sum(B) - dot(A, B))
        dot_prod = np.dot(A, B.T)
        sum_A = np.sum(A, axis=1).reshape(-1, 1)
        sum_B = np.sum(B, axis=1).reshape(1, -1)
        return dot_prod / (sum_A + sum_B - dot_prod + 1e-9)

    K_train = get_tanimoto(X_train, X_train)
    K_test = get_tanimoto(X_test, X_train)
    
    model = SVR(kernel='precomputed', C=C, epsilon=EPS)
    model.fit(K_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(K_test)
