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

    is_log = np.all(y_train >= 0)
    y_train_transformed = np.log1p(y_train) if is_log else y_train

    n_models = 15
    subset_size = int(N_BITS * 0.6)
    ensemble_preds = np.zeros(len(test_smiles))
    
    rng = np.random.default_rng(seed)
    for _ in range(n_models):
        indices = rng.choice(N_BITS, subset_size, replace=False)
        
        X_train_sub = X_train[:, indices]
        X_test_sub = X_test[:, indices]
        
        K_train_sub = tanimoto_kernel(X_train_sub, X_train_sub)
        K_test_sub = tanimoto_kernel(X_test_sub, X_train_sub)
        
        model = SVR(kernel='precomputed', C=C, epsilon=EPSILON)
        model.fit(K_train_sub, y_train_transformed)
        ensemble_preds += model.predict(K_test_sub)

    avg_preds = ensemble_preds / n_models
    return np.expm1(avg_preds) if is_log else avg_preds
