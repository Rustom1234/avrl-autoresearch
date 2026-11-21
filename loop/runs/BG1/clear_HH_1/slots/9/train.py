"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
N_PAIRS = 30  # sampled partners per training molecule
MIN_PAIRS = 100  # below this, fall back to the forest


def pairwise_scores(Xtr, y, Xte, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    I = np.repeat(np.arange(n), N_PAIRS)
    J = rng.integers(0, n, size=n * N_PAIRS)
    keep = y[I] != y[J]
    I, J = I[keep], J[keep]
    if len(I) < MIN_PAIRS:
        return None
    D = Xtr[I] - Xtr[J]
    label = (y[I] > y[J]).astype(int)
    D = np.vstack([D, -D])  # mirror pairs for class balance
    label = np.concatenate([label, 1 - label])
    scale = Xtr.std(axis=0) + 1e-6
    clf = LogisticRegression(fit_intercept=False, C=0.1, max_iter=1000, random_state=seed)
    clf.fit(D / scale, label)
    return (Xte / scale) @ clf.coef_.ravel()


def featurize(smiles, kind):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    if kind == "atompair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
        return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    return np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    ranked = []
    for kind in ("morgan", "atompair", "maccs"):
        Xtr = featurize(train_smiles, kind)
        Xte = featurize(test_smiles, kind)
        score = pairwise_scores(Xtr, y, Xte, seed)
        if score is None:  # fallback: rank-transformed forest
            model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                          min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
            model.fit(Xtr, y)
            score = model.predict(Xte)
        ranked.append(rankdata(score))
    return np.mean(ranked, axis=0) / len(test_smiles)
