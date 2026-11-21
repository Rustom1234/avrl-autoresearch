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
LAMBDA = 1.0
ITERATIONS = 10


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
        y_train = np.log1p(y_train)
    
    # Huber-robust Kernel Ridge Regression via IRLS
    # Solve (K + lambda * W^-1) alpha = y
    W = np.ones((K_train.shape[0], K_train.shape[0]))
    
    for _ in range(ITERATIONS):
        # Solve for alpha: (K + lambda * diag(W)) alpha = y
        # We use a simple diagonal weight matrix for efficiency
        K_reg = K_train + LAMBDA * np.diag(W)
        try:
            alpha = np.linalg.solve(K_reg, y_train)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(K_reg, y_train, rcond=None)[0]
        
        preds_train = K_train @ alpha
        residuals = y_train - preds_train
        med_abs_resid = np.median(np.abs(residuals))
        
        if med_abs_resid > 0:
            delta = 1.345 * med_abs_resid
            W = np.diag(np.minimum(1.0, delta / (np.abs(residuals) + 1e-9)))
    
    preds = K_test @ alpha
    
    if np.all(y_train >= 0) and (y_train + 1 > 0):
        # Check if we should reverse the log1p
        # We need to know the original y_train to decide
        # Since y_train was transformed in place, we check if it's positive
        return np.expm1(preds)
    return preds
