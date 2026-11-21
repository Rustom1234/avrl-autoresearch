"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
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
                         subsample=0.8, colsample_bytree=0.4, reg_lambda=2,
                         objective="reg:pseudohubererror", huber_slope=1.0,
                         n_jobs=8, random_state=seed)
    keep = (Xtr != 0).sum(axis=0) >= 5
    d_idx = np.where(keep)[0]
    d_idx = d_idx[d_idx >= N_BITS]
    with np.errstate(all="ignore"):
        c = np.nan_to_num(np.abs(np.corrcoef(Xtr[:, d_idx].astype(np.float64), rowvar=False)))
    keep[d_idx[np.triu(c, 1).max(axis=0) > 0.95]] = False
    model.fit(Xtr[:, keep], y)
    p_tree = model.predict(Xte[:, keep])

    qt = QuantileTransformer(n_quantiles=min(1000, len(y)), output_distribution="normal", random_state=seed)
    Dtr = qt.fit_transform(Xtr[:, N_BITS:])
    Dte = qt.transform(Xte[:, N_BITS:])
    p_mlp = np.mean([
        MLPRegressor(hidden_layer_sizes=(512, 256), alpha=1e-2, early_stopping=True,
                     max_iter=300, random_state=seed + k).fit(Dtr, y).predict(Dte)
        for k in range(5)], axis=0)
    return 0.5 * p_tree + 0.5 * p_mlp
