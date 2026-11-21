"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES_LGBM = 300
LEARNING_RATE = 0.1
NUM_LEAVES = 31
N_TREES_RF = 100

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
    X_test = featurize(test_smiles)
    
    lgbm = lgb.LGBMClassifier(
        n_estimators=N_TREES_LGBM,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        n_jobs=-1,
        random_state=seed,
        importance_type='gain'
    )
    lgbm.fit(X_train, y_train)
    
    rf = RandomForestClassifier(
        n_estimators=N_TREES_RF,
        max_depth=12,
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(X_train, y_train)
    
    return (lgbm.predict_proba(X_test)[:, 1] + rf.predict_proba(X_test)[:, 1]) / 2
