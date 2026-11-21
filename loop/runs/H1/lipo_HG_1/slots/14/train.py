"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
C = 10.0
EPS = 0.1

def featurize(smiles, use_descriptors=False):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    if not use_descriptors:
        return fps
    
    desc_list = Descriptors.descList
    descs = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: 
            descs.append(np.zeros(len(desc_list)))
            continue
        d = np.array([Descriptors.calc(desc_list[i], mol) for i in range(len(desc_list))])
        descs.append(d)
    
    descs_arr = np.array(descs)
    descs_arr = np.nan_to_num(descs_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Simple clipping for extreme values (0.5th and 99.5th percentiles)
    p05 = np.percentile(descs_arr, 0.5, axis=0)
    p995 = np.percentile(descs_arr, 99.5, axis=0)
    descs_arr = np.clip(descs_arr, p05, p995)
    
    return np.hstack([fps, descs_arr])

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles, use_descriptors=True)
    X_test = featurize(test_smiles, use_descriptors=True)
    
    # Note: Since we added non-binary descriptors, the Tanimoto kernel 
    # is technically no longer the standard Jaccard. However, dot product 
    # is still a common similarity measure for mixed vectors.
    def get_similarity(A, B):
        dot_prod = np.dot(A, B.T)
        sum_A = np.sum(A, axis=1).reshape(-1, 1)
        sum_B = np.sum(B, axis=1).reshape(1, -1)
        return dot_prod / (sum_A + sum_B - dot_prod + 1e-9)

    K_train = get_similarity(X_train, X_train)
    K_test = get_similarity(X_test, X_train)
    
    model = SVR(kernel='precomputed', C=C, epsilon=EPS)
    model.fit(K_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(K_test)
