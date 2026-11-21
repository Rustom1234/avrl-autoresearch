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


ELEMS = [6, 7, 8, 9, 15, 16, 17, 35, 53]


def propagate_feats(mols):
    dim = len(ELEMS) + 6  # element one-hot + "other" slot + 5 atom properties
    out = []
    for m in mols:
        n = m.GetNumAtoms()
        if n == 0:
            out.append(np.zeros(8 * dim, dtype=np.float32))
            continue
        X = np.zeros((n, dim))
        for i, a in enumerate(m.GetAtoms()):
            z = a.GetAtomicNum()
            X[i, ELEMS.index(z) if z in ELEMS else len(ELEMS)] = 1.0
            X[i, len(ELEMS) + 1:] = [float(a.GetDegree()), float(a.GetFormalCharge()),
                                     float(a.GetIsAromatic()), float(a.GetTotalNumHs()),
                                     float(a.IsInRing())]
        A = Chem.GetAdjacencyMatrix(m).astype(float) + np.eye(n)
        P = A / A.sum(1, keepdims=True)
        H = X
        parts = []
        for _ in range(4):
            parts.append(H.sum(0))
            parts.append(H.mean(0))
            H = P @ H
        out.append(np.concatenate(parts))
    return np.array(out, dtype=np.float32)


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    prop = propagate_feats(mols)
    return np.hstack([fp, desc.astype(np.float32), prop])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(featurize(test_smiles))[:, 1]
