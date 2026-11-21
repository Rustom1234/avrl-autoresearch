"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.metrics.pairwise import tanimoto_similarity

# Settings
N_BITS = 2048
RADIUS = 2
ALPHA = 0.5

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, 
        fpSize=N_BITS, 
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    return fps

def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_fps = featurize(train_smiles)
    test_fps = featurize(test_smiles)
    
    # Compute Tanimoto kernel matrix
    K = tanimoto_similarity(train_fps, train_fps)
    
    # Center the target
    y_train = np.asarray(train_y, dtype=np.float64)
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    
    # Solve (K + alpha*I) * alpha = y_centered
    # Using regularization to handle multi-collinearity and over-fitting
    A = K + ALPHA * np.eye(K.shape[0])
    try:
        weights = np.linalg.solve(A, y_centered)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(A, y_centered, rcond=None)[0]
        
    # Prediction: K_test_train * weights + y_mean
    K_test = tanimoto_similarity(test_fps, train_fps)
    return K_test @ weights + y_mean
