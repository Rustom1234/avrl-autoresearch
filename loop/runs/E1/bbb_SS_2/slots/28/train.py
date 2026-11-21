"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def pharm_fp(mol):
    v = np.zeros(N_BITS)
    for b in Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory).GetOnBits():
        v[b % N_BITS] = 1
    return v


ACID_SMARTS = ["[CX3](=O)[OX2H1,OX1-]", "c1nnn[nH]1", "c1nn[nH]n1", "S(=O)(=O)[OX2H1,OX1-]",
               "C(=O)[NH]S(=O)(=O)"]
BASE_SMARTS = ["[NX3;H0;!$(N-[a,C,S]=[O,N,S]);!$(N-a);!$(N-S)]([CX4])([CX4])[CX4]",
               "[NX3;H1;!$(N-[a,C,S]=[O,N,S]);!$(N-a);!$(N-S)]([CX4])[CX4]",
               "[NX3;H2;!$(N-[a,C,S]=[O,N,S]);!$(N-a);!$(N-S)][CX4]"]
ACID_PATS = [Chem.MolFromSmarts(s) for s in ACID_SMARTS]
BASE_PATS = [Chem.MolFromSmarts(s) for s in BASE_SMARTS]


def rule_feats(m):
    tpsa = Descriptors.TPSA(m)
    mw = Descriptors.MolWt(m)
    hbd = Descriptors.NumHDonors(m)
    out = [len(m.GetSubstructMatches(p)) for p in ACID_PATS]
    out += [len(m.GetSubstructMatches(p)) for p in BASE_PATS]
    out += [hbd, tpsa, mw, tpsa / max(mw, 1.0), float(hbd <= 3), float(tpsa < 90), float(mw < 450)]
    return out


def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    R = np.array([rule_feats(m) for m in mols], dtype=np.float64)
    rows = [[f(m) for _, f in Descriptors.descList] for m in mols]
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return np.hstack([X, R, np.array([pharm_fp(m) for m in mols])])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    model = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=8, random_state=seed)
    y = np.asarray(train_y, dtype=np.int64)
    model.fit(Xtr, y)
    p_rf = model.predict_proba(Xte)[:, 1]
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    Qtr = qt.fit_transform(Xtr)
    Qte = qt.transform(Xte)
    p_mlp = np.zeros(len(Xte))
    for i in range(5):
        mlp = MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                            random_state=seed + i, max_iter=200)
        mlp.fit(Qtr, y)
        p_mlp += mlp.predict_proba(Qte)[:, 1] / 5
    return 0.5 * rankdata(p_rf) / len(p_rf) + 0.5 * rankdata(p_mlp) / len(p_mlp)
