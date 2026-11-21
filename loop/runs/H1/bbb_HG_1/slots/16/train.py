"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
LEARNING_RATE = 0.03
NUM_LEAVES = 31
FEATURE_FRACTION = 0.5
SUB_SAMPLE = 0.7
MIN_CHILD_SAMPLES = 10


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)
    X_test = featurize(test_smiles)
    
    probs = []
    for i in range(5):
        model = LGBMClassifier(
            n_estimators=N_TREES,
            learning_rate=LEARNING_RATE,
            num_leaves=NUM_LEAVES,
            feature_fraction=FEATURE_FRACTION,
            subsample=SUB_SAMPLE,
            subsample_freq=1,
            min_child_samples=MIN_CHILD_SAMPLES,
            random_state=seed + i,
            n_jobs=8,
            verbose=-1
        )
        model.fit(X_train, y_train)
        probs.append(model.predict_proba(X_test)[:, 1])
        
    return np.mean(probs, axis=0)
