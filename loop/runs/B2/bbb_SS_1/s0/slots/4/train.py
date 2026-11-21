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
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_LFC = rdMolStandardize.LargestFragmentChooser()
_UNCH = rdMolStandardize.Uncharger()


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    try:
        m2 = _UNCH.uncharge(_LFC.choose(m))
        return m2 if m2 is not None and m2.GetNumAtoms() > 0 else m
    except Exception:
        return m


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows, bits = [], []
    for s in smiles:
        m = standardize(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
        bits.append(fp > 0)
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6).astype(np.float32)
    return X, np.array(bits, dtype=np.float32)


def knn_scores(B_tr, y_tr, B_te, k=15):
    inter = B_te @ B_tr.T
    sim = inter / (B_te.sum(1)[:, None] + B_tr.sum(1)[None, :] - inter + 1e-9)
    idx = np.argsort(-sim, axis=1)[:, :k]
    return y_tr[idx].mean(axis=1)


def member_probs(X, y, w, Xt, B, Bt, seed):
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1),
        CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           random_seed=seed, verbose=0),
    ]
    preds = []
    for m in models:
        m.fit(X, y, sample_weight=w)
        preds.append(m.predict_proba(Xt)[:, 1])
    preds.append(knn_scores(B, y, Bt))
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)
    Z = qt.fit_transform(X)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=300, random_state=seed)
    mlp.fit(Z, y)
    preds.append(mlp.predict_proba(qt.transform(Xt))[:, 1])
    return preds


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X, B = featurize(train_smiles)
    Xt, Bt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    w = np.ones(len(y))
    probs = member_probs(X, y, w, Xt, B, Bt, seed)
    return np.mean([rankdata(p) / len(Xt) for p in probs], axis=0)
