"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys
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


def add_descriptors(train_smiles, test_smiles, Xtr, Xte):
    names = [n for n, _ in Descriptors.descList]

    def calc(smiles):
        rows = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in smiles]
        return np.array([[r[n] for n in names] for r in rows], dtype=np.float64)

    Dtr = np.nan_to_num(calc(train_smiles), nan=0.0, posinf=0.0, neginf=0.0)
    Dte = np.nan_to_num(calc(test_smiles), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(Dtr, [1, 99], axis=0)
    Dtr, Dte = np.clip(Dtr, lo, hi), np.clip(Dte, lo, hi)
    return np.hstack([Xtr, Dtr.astype(np.float32)]), np.hstack([Xte, Dte.astype(np.float32)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    preds = []
    for kind in ("morgan", "atompair", "maccs", "gobbi"):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        Xtr, Xte = featurize(train_smiles, kind), featurize(test_smiles, kind)
        if kind == "morgan":
            Xtr, Xte = add_descriptors(train_smiles, test_smiles, Xtr, Xte)
        model.fit(Xtr, y)
        preds.append(model.predict(Xte))
    return np.mean(preds, axis=0)
