"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
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
N_SVD = 128
KNN_K = 10


_LFC = rdMolStandardize.LargestFragmentChooser()
_UNC = rdMolStandardize.Uncharger()


def standardize(smi):
    m = Chem.MolFromSmiles(smi)
    try:
        s = _UNC.uncharge(_LFC.choose(m))
        if s is not None and s.GetNumAtoms() > 0:
            return s
    except Exception:
        pass
    return m


def tanimoto_knn(B_train, y_train, B_test, k=KNN_K):
    inter = B_test @ B_train.T
    sa = B_test.sum(1)[:, None]
    sb = B_train.sum(1)[None, :]
    sim = inter / np.maximum(sa + sb - inter, 1e-9)
    idx = np.argsort(-sim, axis=1)[:, :k]
    s = np.take_along_axis(sim, idx, axis=1)
    return (s * y_train[idx]).sum(1) / np.maximum(s.sum(1), 1e-9)


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [standardize(s) for s in smiles]
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    counts = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float64)
    bits = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    return desc, counts, bits


def fit_predict(train_smiles, train_y, test_smiles, seed):
    D, C, B = featurize(train_smiles)
    Dt, Ct, Bt = featurize(test_smiles)
    X = np.concatenate([D, C], axis=1).astype(np.float32)
    Xt = np.concatenate([Dt, Ct], axis=1).astype(np.float32)
    y = np.asarray(train_y, dtype=np.int64)
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
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    preds.append(rankdata(tanimoto_knn(B.astype(np.float32), y, Bt.astype(np.float32))) / len(Xt))
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)
    svd = TruncatedSVD(n_components=N_SVD, random_state=seed)
    M = np.concatenate([qt.fit_transform(D), svd.fit_transform(C)], axis=1)
    Mt = np.concatenate([qt.transform(Dt), svd.transform(Ct)], axis=1)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), early_stopping=True, max_iter=500, random_state=seed)
    mlp.fit(M, y)
    preds.append(rankdata(mlp.predict_proba(Mt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
