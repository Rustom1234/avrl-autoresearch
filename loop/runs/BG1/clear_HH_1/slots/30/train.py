"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def scaffold_feats(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        core = list(m.GetSubstructMatch(scaf)) if scaf.GetNumAtoms() else []
        core_set = set(core)
        side = [a.GetIdx() for a in m.GetAtoms() if a.GetIdx() not in core_set]
        f_core = gen.GetCountFingerprintAsNumPy(m, fromAtoms=core) if core else np.zeros(1024)
        f_side = gen.GetCountFingerprintAsNumPy(m, fromAtoms=side) if side else np.zeros(1024)
        out.append(np.concatenate([f_core, f_side]))
    return np.array(out, dtype=np.float32)


def featurize(smiles, kind):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if kind == "scaffold":
        return scaffold_feats(smiles)
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
    for kind in ("morgan", "atompair", "maccs", "gobbi", "scaffold"):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        model.fit(featurize(train_smiles, kind), y)
        preds.append(model.predict(featurize(test_smiles, kind)))
    return np.mean(preds, axis=0)
