"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR

# Settings
N_BITS = 4096
C = 10.0
EPS = 0.1

def get_tanimoto(A, B):
    # A: (n, d), B: (m, d)
    dot_prod = np.dot(A, B.T)
    sum_A = np.sum(A, axis=1).reshape(-1, 1)
    sum_B = np.sum(B, axis=1).reshape(1, -1)
    return dot_prod / (sum_A + sum_B - dot_prod + 1e-9)

def featurize(smiles, radius):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    K_train = np.zeros((len(train_smiles), len(train_smiles)), dtype=np.float32)
    K_test = np.zeros((len(test_smiles), len(train_smiles)), dtype=np.float32)
    
    for r in [1, 2, 3]:
        X_train = featurize(train_smiles, r)
        X_test = featurize(test_smiles, r)
        K_train += get_tanimoto(X_train, X_train)
        K_test += get_tanimoto(X_test, X_train)
    
    K_train /= 3.0
    K_test /= 3.0
    
    model = SVR(kernel='precomputed', C=C, epsilon=EPS)
    model.fit(K_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(K_test)
