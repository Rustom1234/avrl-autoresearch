"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
# Settings
N_BITS = 2048
RADIUS = 2
ALPHA = 0.3


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        
        # Physchem descriptors
        descs = [
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol)
        ]
        fps.append(np.concatenate([fp, np.array(descs, dtype=np.float32)]))
    
    return np.array(fps, dtype=np.float32)


def get_tanimoto_kernel(fps):
    n = fps.shape[0]
    K = np.zeros((n, n))
    # Extract only binary bits for Tanimoto
    bits = fps[:, :N_BITS].astype(bool)
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(bits[i], bits[j]).sum()
            union = np.logical_or(bits[i], bits[j]).sum()
            sim = inter / union if union > 0 else 0.0
            K[i, j] = K[j, i] = sim
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    K = get_tanimoto_kernel(X_train)
    y_centered = y_train - np.mean(y_train)
    
    # Solve (K + alpha*I) * coef = y_centered
    # Add a small ridge to the diagonal for numerical stability
    A = K + ALPHA * np.eye(K.shape[0])
    try:
        coef = np.linalg.solve(A, y_centered)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, y_centered, rcond=None)[0]
    
    X_test = featurize(test_smiles)
    K_test = np.zeros((X_test.shape[0], X_train.shape[0]))
    bits_train = X_train[:, :N_BITS].astype(bool)
    bits_test = X_test[:, :N_BITS].astype(bool)
    
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            inter = np.logical_and(bits_test[i], bits_train[j]).sum()
            union = np.logical_or(bits_test[i], bits_train[j]).sum()
            K_test[i, j] = inter / union if union > 0 else 0.0
            
    return K_test @ coef + np.mean(y_train)
