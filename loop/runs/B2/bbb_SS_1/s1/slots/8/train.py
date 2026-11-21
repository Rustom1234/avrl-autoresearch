"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smi):
    m0 = Chem.MolFromSmiles(smi)
    try:
        m = _UNCHARGER.uncharge(_CHOOSER.Choose(m0))
    except Exception:
        return m0
    return m if m is not None else m0


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = standardize(s)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        rows.append(np.concatenate([desc, fp]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    Xt = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               class_weight='balanced', n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.3, class_weight='balanced', n_jobs=8, random_state=seed, verbose=-1),
        CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           auto_class_weights='Balanced', random_seed=seed, verbose=0),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    # Tanimoto kNN (k=15) on binary Morgan bits, unweighted
    n_desc = len(Descriptors.descList)
    B = (X[:, n_desc:] > 0).astype(np.float32)
    Bt = (Xt[:, n_desc:] > 0).astype(np.float32)
    inter = Bt @ B.T
    sim = inter / np.maximum(Bt.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1e-9)
    nn = np.argsort(-sim, axis=1)[:, :15]
    preds.append(rankdata(y[nn].mean(axis=1)) / len(Xt))
    # MLP on quantile-scaled features; minority class oversampled to parity with a seeded resample
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    minority, majority = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(minority, size=len(majority) - len(minority), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=seed)
    Q = qt.fit_transform(X)
    Qt = qt.transform(Xt)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True, random_state=seed, max_iter=300)
    mlp.fit(Q[idx], y[idx])
    preds.append(rankdata(mlp.predict_proba(Qt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
