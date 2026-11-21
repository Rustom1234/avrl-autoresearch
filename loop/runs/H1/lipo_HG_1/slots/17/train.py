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

from rdkit import Descriptors

def featurize_fp(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def featurize_desc(smiles):
    def get_desc(s):
        m = Chem.MolFromSmiles(s)
        if not m: return np.zeros(7)
        return np.array([Descriptors.MolLogP(m), Descriptors.MolWt(m), 
                         Descriptors.TPSA(m), Descriptors.NumHDonors(m), 
                         Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m), 
                         Descriptors.NumHs(m)])
    return np.array([get_desc(s) for s in smiles], dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train_fp = featurize_fp(train_smiles)
    X_test_fp = featurize_fp(test_smiles)
    X_train_desc = featurize_desc(train_smiles)
    X_test_desc = featurize_desc(test_smiles)
    
    # Standardize descriptors
    mu = np.mean(X_train_desc, axis=0)
    sigma = np.std(X_train_desc, axis=0)
    sigma[sigma == 0] = 1.0
    X_train_desc = (X_train_desc - mu) / sigma
    X_test_desc = (X_test_desc - mu) / sigma
    
    def get_tanimoto(A, B):
        dot_prod = np.dot(A, B.T)
        sum_A = np.sum(A, axis=1).reshape(-1, 1)
        sum_B = np.sum(B, axis=1).reshape(1, -1)
        return dot_prod / (sum_A + sum_B - dot_prod + 1e-9)

    def get_rbf(A, B):
        dist_sq = np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2)
        return np.exp(- (1.0 / X_train_desc.shape[1]) * dist_sq)

    K_tan_train = get_tanimoto(X_train_fp, X_train_fp)
    K_tan_test = get_tanimoto(X_test_fp, X_train_fp)
    K_rbf_train = get_rbf(X_train_desc, X_train_desc)
    K_rbf_test = get_rbf(X_test_desc, X_train_desc)
    
    K_train = K_tan_train + 0.5 * K_rbf_train
    K_test = K_tan_test + 0.5 * K_rbf_test
    
    model = SVR(kernel='precomputed', C=C, epsilon=EPS)
    model.fit(K_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(K_test)
