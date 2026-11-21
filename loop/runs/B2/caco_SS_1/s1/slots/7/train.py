"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
AMINE = Chem.MolFromSmarts('[NX3;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)]')


def featurize(smiles):
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        d = [f(m) for _, f in Descriptors.descList]
        d += [len(m.GetSubstructMatches(ACID)), len(m.GetSubstructMatches(AMINE))]
        c = gen.GetCountFingerprintAsNumPy(m).astype(np.float64)
        rows.append(np.concatenate([np.array(d, dtype=np.float64), c]))
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)


def morgan_bits(smiles):
    from rdkit import DataStructs
    out = np.zeros((len(smiles), N_BITS), dtype=np.float64)
    for i, s in enumerate(smiles):
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), RADIUS, nBits=N_BITS)
        arr = np.zeros((N_BITS,), dtype=np.float64)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def knn_predict(train_smiles, y, test_smiles, k=7):
    A = morgan_bits(test_smiles)
    B = morgan_bits(train_smiles)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    dist = 1.0 - inter / np.maximum(union, 1e-9)
    idx = np.argsort(dist, axis=1)[:, :k]
    d = np.take_along_axis(dist, idx, axis=1)
    w = 1.0 / np.maximum(d, 1e-6)
    return (w * y[idx]).sum(1) / w.sum(1)


def blend_fit_predict(X_a, y_a, smi_a, X_b, smi_b, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(objective='huber', n_estimators=300, learning_rate=0.03, num_leaves=15,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                              min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    model.fit(X_a, y_a)
    et = model.predict(X_b)
    knn = knn_predict(smi_a, y_a, smi_b)
    return 0.75 * et + 0.25 * knn


def fit_predict(train_smiles, train_y, test_smiles, seed):
    from sklearn.model_selection import KFold
    from scipy.optimize import minimize
    y = np.asarray(train_y, dtype=np.float64)
    nd = len(Descriptors.descList) + 2
    X_tr = featurize(train_smiles)[:, :nd]
    X_te = featurize(test_smiles)[:, :nd]
    smi_tr = list(train_smiles)
    smi_te = list(test_smiles)

    # Out-of-fold blended predictions on the training molecules
    oof = np.zeros(len(y))
    for tr, va in KFold(n_splits=4, shuffle=True, random_state=seed).split(X_tr):
        oof[va] = blend_fit_predict(X_tr[tr], y[tr], [smi_tr[i] for i in tr],
                                    X_tr[va], [smi_tr[i] for i in va], seed)

    # Least-absolute-deviation line y = a + b * pred
    def l1(p):
        return np.mean(np.abs(y - (p[0] + p[1] * oof)))
    a, b = minimize(l1, x0=np.array([0.0, 1.0]), method='Nelder-Mead').x

    pred = blend_fit_predict(X_tr, y, smi_tr, X_te, smi_te, seed)
    out = a + b * pred
    return np.clip(out, y.min(), y.max())
