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
C_GRID = [1.0, 3.0, 10.0, 30.0]
EPS_GRID = [0.05, 0.2]

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    
    def get_tanimoto(A, B):
        dot_prod = np.dot(A, B.T)
        sum_A = np.sum(A, axis=1).reshape(-1, 1)
        sum_B = np.sum(B, axis=1).reshape(1, -1)
        return dot_prod / (sum_A + sum_B - dot_prod + 1e-9)

    K_train = get_tanimoto(X_train, X_train)
    K_test = get_tanimoto(X_test, X_train)
    
    all_preds = []
    for C in C_GRID:
        for EPS in EPS_GRID:
            model = SVR(kernel='precomputed', C=C, epsilon=EPS)
            model.fit(K_train, np.asarray(train_y, dtype=np.float64))
            all_preds.append(model.predict(K_test))
            
    return np.mean(all_preds, axis=0)
