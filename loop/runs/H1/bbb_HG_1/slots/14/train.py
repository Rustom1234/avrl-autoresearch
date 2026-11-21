"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMClassifier
import pandas as pd

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

def get_descriptors(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    data = []
    for mol in mols:
        if mol is None: continue
        desc = []
        for name in Descriptors.descList:
            try:
                val = getattr(mol, name)(
                )
                desc.append(val if np.isfinite(val) else 0)
            except: desc.append(0)
        data.append(desc)
    return np.array(data, dtype=np.float32)

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols if m is not None], dtype=np.float32)
    # Ensure we handle missing mols consistently; simplest is to assume all valid
    # For this task, let's assume valid SMILES
    desc = get_descriptors(smiles)
    # Clip descriptors to 1st and 99th percentiles
    p1, p99 = np.percentile(desc, [1, 99], axis=0)
    desc = np.clip(desc, p1, p99)
    
    # We need to match the shape of fps. If a mol was None, fps and desc shapes differ.
    # To keep it simple and robust:
    fps = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    desc = get_descriptors(smiles)
    p1, p99 = np.percentile(desc, [1, 99], axis=0)
    desc = np.clip(desc, p1, p99)
    return np.hstack([desc, fps])

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
