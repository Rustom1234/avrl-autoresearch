"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import xgboost as xgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
LEARNING_RATE = 0.1

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    desc = np.array([
        [Descriptors.MolWt(m),
         Descriptors.MolLogP(m),
         Descriptors.NumHDonors(m),
         Descriptors.NumHAcceptors(m),
         Descriptors.TPSA(m)] for m in mols], dtype=np.float32)
    
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)
    
    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        max_depth=6,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict_proba(featurize(test_smiles))[:, 1]
