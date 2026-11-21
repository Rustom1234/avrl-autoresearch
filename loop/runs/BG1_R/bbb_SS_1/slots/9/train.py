"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


ACID = Chem.MolFromSmarts("[$([CX3](=O)[OX2H1,OX1-]),$([SX4,PX4](=O)[OX2H1,OX1-]),$(c1nn[nH]n1)]")
BASE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N#*);!$(N=*)]")


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        pos = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() > 0)
        neg = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() < 0)
        rows.append([
            Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.MolLogP(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
            Descriptors.NumRotatableBonds(m), Descriptors.FractionCSP3(m),
            pos, neg, Descriptors.RingCount(m),
            len(m.GetSubstructMatches(ACID)), len(m.GetSubstructMatches(BASE)),
            Descriptors.NumAromaticRings(m), m.GetNumHeavyAtoms(),
        ])
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    models = [
        LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.7, n_jobs=8, random_state=seed, verbose=-1),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
