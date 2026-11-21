"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR
import scipy.linalg as la

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

    K_train = tanimoto_kernel(X_train, X_train)
    K_test = tanimoto_kernel(X_test, X_train)

    if np.all(y_train >= 0):
        y_train_transformed = np.log1p(y_train)
    else:
        y_train_transformed = y_train

    best_ll = -np.inf
    best_s2, best_sigma2 = 1.0, 0.1

    s2_grid = [0.5, 1.0, 2.0]
    sigma2_grid = [0.05, 0.1, 0.2, 0.4]

    for s2 in s2_grid:
        for sigma2 in sigma2_grid:
            K_full = s2 * K_train + sigma2 * np.eye(K_train.shape[0])
            try:
                L = la.cholesky(K_full, lower=True)
                alpha = la.solve(L.T, la.solve(L, y_train_transformed, lower=True).T, lower=False).flatten()
                # Log Marginal Likelihood
                # Formula: -0.5 * y^T * alpha - sum(log(diag(L))) - n/2 * log(2*pi)
                ll = -0.5 * np.dot(y_train_transformed, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * len(y_train_transformed) * np.log(2 * np.pi)
                if ll > best_ll:
                    best_ll = ll
                    best_s2, best_sigma2 = s2, sigma2
            except la.LinAlgError:
                continue

    K_final = best_s2 * K_train + best_sigma2 * np.eye(K_train.shape[0])
    L = la.cholesky(K_final, lower=True)
    alpha = la.solve(L.T, la.solve(L, y_train_transformed, lower=True).T, lower=False).flatten()
    
    # Test prediction: K_test * alpha
    preds = np.dot(K_test, alpha)
    
    if np.all(y_train >= 0):
        return np.expm1(preds)
    else:
        return preds
