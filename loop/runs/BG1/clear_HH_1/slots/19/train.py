"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


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
    n_train = len(train_smiles)
    all_smiles = list(train_smiles) + list(test_smiles)
    # Unsupervised transductive SVD over train + test Morgan bits (no labels used)
    morgan_all = featurize(all_smiles, "morgan")
    keep = morgan_all.sum(axis=0) >= 2
    svd_scores = TruncatedSVD(n_components=64, random_state=seed).fit_transform(morgan_all[:, keep])
    morgan_all = np.hstack([morgan_all, svd_scores.astype(np.float32)])
    preds = []
    for kind in ("morgan", "atompair", "maccs"):
        X_all = morgan_all if kind == "morgan" else featurize(all_smiles, kind)
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        model.fit(X_all[:n_train], y)
        preds.append(model.predict(X_all[n_train:]))
    return np.mean(preds, axis=0)
