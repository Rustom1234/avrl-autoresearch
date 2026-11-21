"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from rdkit.Chem import Descriptors

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


DESC_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
              "NumAromaticRings", "FractionCSP3", "RingCount", "HeavyAtomCount", "NumAliphaticRings",
              "NumSaturatedRings", "NumHeteroatoms", "LabuteASA", "MolMR", "NumValenceElectrons",
              "BertzCT", "HallKierAlpha", "Kappa1", "Kappa2"]
MONO = [1 if n == "MolLogP" else 0 for n in DESC_NAMES]


def descriptors(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    X = np.array([[getattr(Descriptors, n)(m) for n in DESC_NAMES] for m in mols], dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    preds = []
    for kind in ("morgan", "atompair", "maccs", "gobbi"):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        model.fit(featurize(train_smiles, kind), y)
        preds.append(model.predict(featurize(test_smiles, kind)))
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                        l2_regularization=1.0, monotonic_cst=MONO, random_state=seed)
    hgb.fit(descriptors(train_smiles), y)
    return 0.5 * np.mean(preds, axis=0) + 0.5 * hgb.predict(descriptors(test_smiles))
