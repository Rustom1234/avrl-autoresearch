"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, rdFingerprintGenerator
from collections import Counter
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


def brics_frags(smiles):
    sets = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        sets.append(set(BRICS.BRICSDecompose(m)) if m is not None else set())
    return sets


def brics_vocab(frag_sets):
    counts = Counter(f for fs in frag_sets for f in fs)
    kept = sorted(((f, c) for f, c in counts.items() if c >= 5), key=lambda fc: (-fc[1], fc[0]))[:512]
    return {f: i for i, (f, _) in enumerate(kept)}


def brics_matrix(frag_sets, vocab):
    X = np.zeros((len(frag_sets), len(vocab)), dtype=np.float32)
    for r, fs in enumerate(frag_sets):
        for f in fs:
            j = vocab.get(f)
            if j is not None:
                X[r, j] = 1.0
    return X


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    train_frags = brics_frags(train_smiles)
    vocab = brics_vocab(train_frags)
    X_train = np.hstack([featurize(train_smiles), brics_matrix(train_frags, vocab)])
    model.fit(X_train, np.asarray(train_y, dtype=np.int64))
    X_test = np.hstack([featurize(test_smiles), brics_matrix(brics_frags(test_smiles), vocab)])
    return model.predict_proba(X_test)[:, 1]
