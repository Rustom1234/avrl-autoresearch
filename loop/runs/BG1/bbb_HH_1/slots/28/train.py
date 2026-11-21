"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
MIN_DF = 5


def featurize(mols, counts, vocab):
    fp = np.zeros((len(mols), len(vocab)), dtype=np.float32)
    for i, c in enumerate(counts):
        for k, v in c.items():
            j = vocab.get(k)
            if j is not None:
                fp[i, j] = v
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return np.hstack([fp, desc.astype(np.float32)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=2**20)
    train_mols = [Chem.MolFromSmiles(s) for s in train_smiles]
    test_mols = [Chem.MolFromSmiles(s) for s in test_smiles]
    train_counts = [gen.GetSparseCountFingerprint(m).GetNonzeroElements() for m in train_mols]
    test_counts = [gen.GetSparseCountFingerprint(m).GetNonzeroElements() for m in test_mols]
    df = Counter(k for c in train_counts for k in c)
    vocab = {k: j for j, k in enumerate(sorted(k for k, n in df.items() if n >= MIN_DF))}
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_mols, train_counts, vocab), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_mols, test_counts, vocab))[:, 1]
