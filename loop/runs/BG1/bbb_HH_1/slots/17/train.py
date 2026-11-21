"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

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


def ring_keys(smiles):
    # Key = sorted fused ring-system fragments (side chains and linkers removed) of each molecule.
    keys = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        systems = []
        for ring in m.GetRingInfo().AtomRings():
            atoms = set(ring)
            for old in [x for x in systems if x & atoms]:
                systems.remove(old)
                atoms |= old
            systems.append(atoms)
        keys.append(".".join(sorted(Chem.MolFragmentToSmiles(m, atomsToUse=sorted(x)) for x in systems)))
    arr = np.empty(len(keys), dtype=object)
    arr[:] = keys
    return arr


def te_fit(keys, y):
    stats = {}
    for k, t in zip(keys, y):
        pos, n = stats.get(k, (0, 0))
        stats[k] = (pos + int(t), n + 1)
    return stats


def te_apply(stats, keys, prior, m=10.0):
    # Columns: smoothed positive rate (unseen -> prior), and number of training molecules with this key.
    out = np.zeros((len(keys), 2), dtype=np.float32)
    for i, k in enumerate(keys):
        pos, n = stats.get(k, (0, 0))
        out[i] = [(pos + m * prior) / (n + m), n]
    return out


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    X_tr, X_te = featurize(train_smiles), featurize(test_smiles)
    k_tr, k_te = ring_keys(train_smiles), ring_keys(test_smiles)
    te_tr = np.zeros((len(y), 2), dtype=np.float32)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(X_tr):
        te_tr[b] = te_apply(te_fit(k_tr[a], y[a]), k_tr[b], y[a].mean())
    te_te = te_apply(te_fit(k_tr, y), k_te, y.mean())
    X_tr = np.hstack([X_tr, te_tr])
    X_te = np.hstack([X_te, te_te])
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    model.fit(X_tr, y)
    return model.predict_proba(X_te)[:, 1]
