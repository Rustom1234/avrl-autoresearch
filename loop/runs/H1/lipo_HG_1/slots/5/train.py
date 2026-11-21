"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.15


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    
    best_leaf = 3
    best_score = float('inf')
    
    # Use a better default and remove the problematic inner loop to save time
    best_leaf = 10
    
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, 
                                  min_samples_leaf=best_leaf, max_samples=0.8, 
                                  n_jobs=-1, random_state=seed)
    model.fit(X, y)
    return model.predict(featurize(test_smiles))
