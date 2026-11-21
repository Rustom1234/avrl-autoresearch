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
        X_train = featurize(train_smiles, kind)
        X_test = featurize(test_smiles, kind)
        kind_preds = []
        for j, mf in enumerate((0.05, 0.15, 0.3, 0.6)):
            model = RandomForestRegressor(n_estimators=75, max_features=mf,
                                          min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1,
                                          random_state=seed * 4 + j)
            model.fit(X_train, y)
            kind_preds.append(model.predict(X_test))
        preds.append(np.mean(kind_preds, axis=0))
    return np.mean(preds, axis=0)
