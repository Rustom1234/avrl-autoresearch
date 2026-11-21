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
C = 5.0
EPSILON = 0.1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    
    # Compute Tanimoto kernel: K(i,j) = sum(min(Ii, Ij)) / sum(max(Ii, Ij))
    # For binary fingerprints: K = sum(i & j) / sum(i | j)
    # Note: sum(max) = sum(x1) + sum(x2) - sum(min)
    dot_train = np.dot(X_train, X_train.T)
    sum_train = np.sum(X_train, axis=1).reshape(-1, 1)
    K_train = dot_train / (sum_train + sum_train.T - dot_train + 1e-9)
    
    dot_test = np.dot(X_train, X_test.T)
    sum_test = np.sum(X_test, axis=1).reshape(1, -1)
    K_test = dot_test / (sum_train + sum_test - dot_test + 1e-9)
    
    model = SVR(kernel='precomputed', C=C, epsilon=EPSILON)
    model.fit(K_train, np.asarray(train_y, dtype=np.float64))
    
    # sklearn SVR with precomputed kernel expects K(test, train)
    return model.predict(K_test.T)
