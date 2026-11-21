"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADII = [1, 2, 3]
N_TREES = 600
LEARNING_RATE = 0.03
NUM_LEAVES = 31
FEATURE_FRACTION = 0.3
BAGGING_FRACTION = 0.8
BAGGING_FREQ = 1
MIN_CHILD_SAMPLES = 10


def featurize(smiles):
    all_fps = []
    for r in RADII:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=N_BITS, useFeatures=True)
        all_fps.append([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles])
    return np.concatenate(all_fps, axis=1).astype(np.float32)


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
