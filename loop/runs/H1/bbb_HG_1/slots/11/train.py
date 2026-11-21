"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 600
LEARNING_RATE = 0.03
NUM_LEAVES = 31
FEATURE_FRACTION = 0.3
BAGGING_FRACTION = 0.8
BAGGING_FREQ = 1
MIN_CHILD_SAMPLES = 10


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    maccs = []
    morgan = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        maccs.append(DataStructs.GetMorganFingerprintAsNumPy(mol, 2, nBits=2048)) # Wait, MACCS is different
    # Correct implementation for MACCS + Morgan
    maccs = []
    morgan = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        maccs.append(DataStructs.GetMACCSKeysAsNumPy(mol))
        morgan.append(gen.GetFingerprintAsNumPy(mol))
    return np.hstack([np.array(maccs), np.array(morgan)]).astype(np.float32)

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        feats.append(gen.GetFingerprintAsNumPy(mol))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        feature_fraction=FEATURE_FRACTION,
        bagging_fraction=BAGGING_FRACTION,
        bagging_freq=BAGGING_FREQ,
        min_child_samples=MIN_CHILD_SAMPLES,
        random_state=seed,
        n_jobs=8,
        verbose=-1
    )
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
