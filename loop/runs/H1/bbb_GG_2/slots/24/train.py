"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 3
N_TREES = 300
LEARNING_RATE = 0.1
NUM_LEAVES = 31

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, count=True)
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
    model = lgb.LGBMClassifier(
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        n_jobs=-1,
        random_state=seed,
        importance_type='gain'
    )
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
