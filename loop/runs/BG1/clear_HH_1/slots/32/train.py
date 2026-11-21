"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors
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


def physchem(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return np.array([[Descriptors.MolLogP(m), Descriptors.TPSA(m), Descriptors.MolWt(m)] for m in mols],
                    dtype=np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    aux = physchem(train_smiles)
    aux = (aux - aux.mean(axis=0)) / (aux.std(axis=0) + 1e-9)
    target = np.column_stack([y, 0.5 * aux])
    preds = []
    for kind in ("morgan", "atompair", "maccs", "gobbi"):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        model.fit(featurize(train_smiles, kind), target)
        preds.append(model.predict(featurize(test_smiles, kind))[:, 0])
    return np.mean(preds, axis=0)
