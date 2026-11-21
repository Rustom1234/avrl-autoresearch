"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 700
LEARNING_RATE = 0.03
SVR_C = 5.0
SVR_EPS = 0.1

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, 
        fpSize=N_BITS, 
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Morgan fingerprints
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    # RDKit descriptors
    desc = np.array([[Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m), 
                               Descriptors.NumHAcceptors(m), Descriptors.TPSA(m)] 
                      for m in mols], dtype=np.float32)
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    X_test = featurize(test_smiles)
    
    # Compute Tanimoto kernel
    # Morgan fingerprints are bit vectors, so Tanimoto = (A & B) / (A | B)
    # We use dot product on bit vectors to get intersection count for speed
    def get_tanimoto(A, B):
        dot = np.dot(A, B.T)
        sq_norm = np.sum(A**2, axis=1).reshape(-1, 1) + np.sum(B**2, axis=1).reshape(1, -1)
        return dot / (sq_norm + 1e-7)

    K_train = get_tanimoto(X_train, X_train)
    K_test = get_tanimoto(X_train, X_test)
    
    # SVR with precomputed Tanimoto kernel
    # To keep it efficient, we only use the fingerprint bits (first 2048 columns)
    # if there were more columns, we'd slice them here.
    svr = SVR(kernel='precomputed', C=SVR_C, epsilon=SVR_EPS)
    svr.fit(K_train, y_train)
    
    return svr.predict(K_test)
