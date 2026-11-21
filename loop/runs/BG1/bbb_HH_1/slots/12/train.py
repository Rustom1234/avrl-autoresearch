"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def bits(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)


def max_sim(query, ref, ref_y):
    inter = query @ ref.T
    sim = inter / (query.sum(1)[:, None] + ref.sum(1)[None, :] - inter + 1e-9)
    pos = sim[:, ref_y == 1].max(1) if (ref_y == 1).any() else np.zeros(len(query))
    neg = sim[:, ref_y == 0].max(1) if (ref_y == 0).any() else np.zeros(len(query))
    return np.column_stack([pos, neg, pos - neg]).astype(np.float32)


def nn_features(train_smiles, train_y, test_smiles):
    X_tr, X_te = bits(train_smiles), bits(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    _, groups = np.unique(np.array(scaf, dtype=object), return_inverse=True)
    F_tr = np.zeros((len(y), 3), dtype=np.float32)
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(X_tr, y, groups):
        F_tr[va_idx] = max_sim(X_tr[va_idx], X_tr[tr_idx], y[tr_idx])
    return F_tr, max_sim(X_te, X_tr, y)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    F_tr, F_te = nn_features(train_smiles, train_y, test_smiles)
    model.fit(np.hstack([featurize(train_smiles), F_tr]), np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(np.hstack([featurize(test_smiles), F_te]))[:, 1]
