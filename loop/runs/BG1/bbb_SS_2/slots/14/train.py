"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import rankdata
from rdkit.Chem import Descriptors
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def descriptors(smiles):
    names = [d[0] for d in Descriptors._descList]
    X = np.array([[f(Chem.MolFromSmiles(s)) for _, f in Descriptors._descList] for s in smiles], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    Ftr, Fte = featurize(train_smiles), featurize(test_smiles)
    Dtr, Dte = descriptors(train_smiles), descriptors(test_smiles)
    preds = []

    rf = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                n_jobs=-1, random_state=seed)
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in train_smiles]
    _, inv, cnt = np.unique(scaf, return_inverse=True, return_counts=True)
    sw = 1.0 / np.sqrt(cnt[inv])
    rf.fit(np.hstack([Dtr, Ftr]), y, sample_weight=sw)
    preds.append(rf.predict_proba(np.hstack([Dte, Fte]))[:, 1])

    svm = SVC(kernel="precomputed", C=5.0, random_state=seed)
    svm.fit(tanimoto(Ftr, Ftr), y)
    preds.append(svm.decision_function(tanimoto(Fte, Ftr)))

    gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.5, random_state=seed, n_jobs=8, verbose=-1)
    gb.fit(Dtr, y)
    preds.append(gb.predict_proba(Dte)[:, 1])

    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)
