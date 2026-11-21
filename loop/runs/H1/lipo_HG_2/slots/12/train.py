"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
C_PARAM = 5.0
EPS_PARAM = 0.1


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


def tanimoto_kernel(A, B):
    # A: (N, BITS), B: (M, BITS)
    dot_product = np.dot(A, B.T)
    sum_A = np.sum(A, axis=1).reshape(-1, 1)
    sum_B = np.sum(B, axis=1).reshape(1, -1)
    return dot_product / (sum_A + sum_B - dot_product + 1e-8)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train_full = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    
    X_train_bits = X_train_full[:, :N_BITS]
    
    # Compute training kernel
    K_train = tanimoto_kernel(X_train_bits, X_train_bits)
    
    # SVR with precomputed Tanimoto kernel
    model = SVR(kernel='precomputed', C=C_PARAM, epsilon=EPS_PARAM)
    model.fit(K_train, y_train)
    
    X_test_full = featurize(test_smiles)
    X_test_bits = X_test_full[:, :N_BITS]
    
    # Compute test-to-train kernel
    K_test = tanimoto_kernel(X_test_bits, X_train_bits)
    
    return model.predict(K_test)
