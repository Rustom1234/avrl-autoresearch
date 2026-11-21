"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, Descriptors, Crippen, Lipinski, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


DESC_FUNCS = [
    Crippen.MolLogP, Crippen.MolMR, rdMolDescriptors.CalcTPSA, Descriptors.MolWt,
    Lipinski.NumHDonors, Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds,
    rdMolDescriptors.CalcFractionCSP3, rdMolDescriptors.CalcNumAromaticRings,
    rdMolDescriptors.CalcNumRings, rdMolDescriptors.CalcNumAliphaticRings,
    rdMolDescriptors.CalcNumAromaticCarbocycles, rdMolDescriptors.CalcNumAromaticHeterocycles,
    rdMolDescriptors.CalcNumHeteroatoms, rdMolDescriptors.CalcNumAtomStereoCenters,
    rdMolDescriptors.CalcLabuteASA, Descriptors.HeavyAtomCount, Descriptors.NumValenceElectrons,
    Descriptors.NHOHCount, Descriptors.NOCount, Descriptors.FpDensityMorgan1,
    Descriptors.qed, Descriptors.BertzCT,
]


def featurize(smiles, kind):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if kind == "desc":
        X = np.array([[f(m) for f in DESC_FUNCS] for m in mols], dtype=np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    if kind == "atompair":
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
        return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    return np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    preds = []
    for kind in ("desc",):
        model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=-1, random_state=seed)
        model.fit(featurize(train_smiles, kind), y)
        preds.append(model.predict(featurize(test_smiles, kind)))
    return np.mean(preds, axis=0)
