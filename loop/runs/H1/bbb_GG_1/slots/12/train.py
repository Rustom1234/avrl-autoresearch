"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)
    
    # Bagging ensemble of 3 Random Forests with different max_features
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=0.6, random_state=seed),
        RandomForestClassifier(n_estimators=N_TREES, max_features=0.7, random_state=seed + 1),
        RandomForestClassifier(n_estimators=N_TREES, max_features=0.8, random_state=seed + 2)
    ]
    
    probs = np.zeros((len(test_smiles),))
    for model in models:
        model.fit(X_train, y_train)
        probs += model.predict_proba(X_test)[:, 1]
        
    return probs / 3.0
