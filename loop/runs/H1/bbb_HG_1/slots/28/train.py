"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from catboost import CatBoostClassifier
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Add basic descriptors: MW, LogP, TPSA
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)])
        features.append(np.concatenate([fp, desc]))
    return np.array(features, dtype=np.float32)

from sklearn.neighbors import NearestNeighbors

def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_feats = featurize(train_smiles)
    test_feats = featurize(test_smiles)
    
    # Use Morgan fingerprints for KNN
    # Since rdFingerprintGenerator returns NumPy arrays, we use cosine similarity
    # which is equivalent to Tanimoto for bit vectors of the same length.
    knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    knn.fit(train_feats)
    
    distances, indices = knn.kneighbors(test_feats)
    
    # distances are 1 - Tanimoto
    # weights = similarity^3 = (1 - distances)^3
    similarities = 1.0 - distances
    weights = np.power(similarities, 3)
    
    # Weighted average of train_y
    train_y_arr = np.asarray(train_y, dtype=np.float64)
    preds = np.zeros(len(test_smiles))
    for i in range(len(test_smiles)):
        neighbor_indices = indices[i]
        neighbor_weights = weights[i]
        preds[i] = np.sum(neighbor_weights * train_y_arr[neighbor_indices]) / np.sum(neighbor_weights)
        
    return preds
