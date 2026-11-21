"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return np.hstack([fp, desc.astype(np.float32)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    names = [n for n, _ in Descriptors.descList]
    decreasing = {"TPSA", "NumHDonors", "NumHAcceptors", "MolWt", "NumRotatableBonds"}
    desc_signs = [-1 if n in decreasing else (1 if n == "MolLogP" else 0) for n in names]
    constraints = [0] * N_BITS + desc_signs
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                           monotone_constraints=constraints, monotone_constraints_method="advanced",
                           n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
