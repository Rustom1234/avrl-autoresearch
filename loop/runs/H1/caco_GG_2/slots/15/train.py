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
MAX_FEATURES_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    
    best_score = float('inf')
    best_mf = MAX_FEATURES_LIST[2]
    
    # Use a simple internal split to select best MAX_FEATURES
    indices = np.arange(len(X))
    np.random.default_rng(seed).shuffle(indices)
    split = int(len(X) * 0.8)
    
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    for mf in MAX_FEATURES_LIST:
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=mf, min_samples_leaf=MIN_SAMPLES_LEAF,
                                      n_jobs=-1, random_state=seed)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        score = np.mean(np.abs(preds - y[val_idx]))
        if score < best_score:
            best_score = score
            best_mf = mf
            
    final_model = RandomForestRegressor(n_estimators=N_TREES, max_features=best_mf, min_samples_leaf=MIN_SAMPLES_LEAF,
                                         n_jobs=-1, random_state=seed)
    final_model.fit(X, y)
    return final_model.predict(featurize(test_smiles))
