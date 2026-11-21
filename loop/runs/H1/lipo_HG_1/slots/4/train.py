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
    
    # Small inner search for min_samples_leaf
    for leaf in [3, 5, 10]:
        # Simple 3-fold split for hyperparameter search
        indices = np.arange(len(X))
        np.random.shuffle(indices, random_state=seed)
        folds = np.split(indices, 3)
        
        fold_scores = []
        for i in range(3):
            train_idx = np.concatenate([folds[j] for j in range(3) if j != i])
            val_idx = folds[i]
            
            m = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, 
                                      min_samples_leaf=leaf, max_samples=0.7, 
                                      n_jobs=-1, random_state=seed)
            m.fit(X[train_idx], y[train_idx])
            score = np.mean(np.abs(m.predict(X[val_idx]) - y[val_idx]))
            fold_scores.append(score)
        
        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_leaf = leaf
            
    # Final fit
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, 
                                  min_samples_leaf=best_leaf, max_samples=0.7, 
                                  n_jobs=-1, random_state=seed)
    model.fit(X, y)
    return model.predict(featurize(test_smiles))
