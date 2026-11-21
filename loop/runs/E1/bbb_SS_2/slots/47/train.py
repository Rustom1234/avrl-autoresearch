"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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


BASIC_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O)]")
ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
HALOGEN = Chem.MolFromSmarts("[F,Cl,Br,I]")


def compact_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Descriptors.TPSA(m), Descriptors.MolLogP(m), Descriptors.MolWt(m),
                     Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                     Descriptors.NumRotatableBonds(m), Chem.GetFormalCharge(m),
                     len(m.GetSubstructMatches(BASIC_AMINE)), len(m.GetSubstructMatches(ACID)),
                     Descriptors.NumAromaticRings(m), Descriptors.FractionCSP3(m),
                     len(m.GetSubstructMatches(HALOGEN))])
    return np.array(rows, dtype=np.float64)


def featurize(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    rows = [[f(m) for _, f in Descriptors.descList] for m in mols]
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return np.hstack([X, np.array([pharm_fp(m) for m in mols])])


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
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                         random_state=seed)
    hgb.fit(compact_features(train_smiles), y)
    p_hgb = hgb.predict_proba(compact_features(test_smiles))[:, 1]
    n = len(p_rf)
    return (rankdata(p_rf) + rankdata(p_mlp) + rankdata(p_hgb)) / (3 * n)
