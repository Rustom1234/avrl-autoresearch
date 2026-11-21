"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from xgboost import XGBRegressor

# Settings
N_BITS = 4096
RADIUS = 3
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1

SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[CX3](=O)[OX1H0-,OX2H1]",  # carboxylic acid
    "[SX4](=O)(=O)[OX2H1,OX1-]",  # sulfonic acid
    "c1nn[nH]n1",  # tetrazole
    "c1nnn[nH]1",  # tetrazole tautomer
    "[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N#*);!$(N=*);!$(N-N);!$(N-O)]([CX4])",  # aliphatic amine
    "[NX3][CX3]=[NX2;!a]",  # amidine/guanidine
    "[NX3][CX3](=[NX2])[NX3]",  # guanidine
    "[NX4+]",  # quaternary N
    "c[OX2H1]",  # phenol
]]


def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mg = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    morgan = np.array([mg.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6).astype(np.float32)
    ion = np.array([[len(m.GetSubstructMatches(p)) for p in SMARTS] + [Chem.GetFormalCharge(m)]
                    for m in mols], dtype=np.float32)
    return np.hstack([morgan, desc, ion])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    model = XGBRegressor(tree_method="hist", n_estimators=800, learning_rate=0.03, max_depth=6,
                         subsample=0.8, colsample_bytree=0.2, reg_lambda=2,
                         objective="reg:pseudohubererror", huber_slope=1.0,
                         n_jobs=8, random_state=seed)
    model.fit(Xtr, y)
    return model.predict(Xte)
