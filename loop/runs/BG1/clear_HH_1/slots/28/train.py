"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from scipy.stats import rankdata
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
    if kind == "gobbi":
        out = np.zeros((len(mols), N_BITS), dtype=np.float32)
        for i, m in enumerate(mols):
            on = np.array(list(Generate.Gen2DFingerprint(m, Gobbi_Pharm2D.factory).GetOnBits()), dtype=int)
            out[i, on % N_BITS] = 1
        return out
    return np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    preds = []
    for kind in ("morgan", "atompair", "maccs", "gobbi"):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        model.fit(featurize(train_smiles, kind), y)
        preds.append(model.predict(featurize(test_smiles, kind)))
    # proximity-kNN member: leaf co-occurrence of a Morgan forest picks 15 nearest training molecules
    X_tr = featurize(train_smiles, "morgan")
    X_te = featurize(test_smiles, "morgan")
    forest = RandomForestRegressor(n_estimators=500, max_features=MAX_FEATURES,
                                   min_samples_leaf=5, n_jobs=-1, random_state=seed)
    forest.fit(X_tr, y)
    leaves_tr = forest.apply(X_tr)
    leaves_te = forest.apply(X_te)
    prox = np.zeros((len(test_smiles), len(train_smiles)))
    for t in range(leaves_tr.shape[1]):
        prox += leaves_te[:, t][:, None] == leaves_tr[:, t][None, :]
    prox /= leaves_tr.shape[1]
    top = np.argsort(-prox, axis=1)[:, :15]
    w = np.take_along_axis(prox, top, axis=1) + 1e-12
    preds.append((w * y[top]).sum(axis=1) / w.sum(axis=1))
    return np.mean(preds, axis=0)
