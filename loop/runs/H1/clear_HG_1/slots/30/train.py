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
EPSILON = 0.05


def get_fp(smiles, radius=2, is_morgan=True):
    if is_morgan:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=N_BITS)
    else:
        gen = rdFingerprintGenerator.GetAtomPairFingerprintGenerator(fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def tanimoto_kernel(X1, X2):
    dot = np.dot(X1, X2.T)
    sum_sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
    sum_sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
    return dot / (sum_sq1 + sum_sq2 - dot + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y_train = np.asarray(train_y, dtype=np.float64)
    
    kernels_train = []
    kernels_test = []
    
    # Morgan fingerprints for radii 1, 2, 3
    for r in [1, 2, 3]:
        X_train = get_fp(train_smiles, radius=r, is_morgan=True)
        X_test = get_fp(test_smiles, radius=r, is_morgan=True)
        kernels_train.append(tanimoto_kernel(X_train, X_train))
        kernels_test.append(tanimoto_kernel(X_test, X_train))
        
    K_train = np.mean(kernels_train, axis=0)
    K_test = np.mean(kernels_test, axis=0)

    model = SVR(kernel='precomputed', C=C, epsilon=EPSILON)

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
        model.fit(K_train, y_train_transformed)
        preds = model.predict(K_test)
        return np.expm1(preds)
    else:
        model.fit(K_train, y_train)
        return model.predict(K_test)
