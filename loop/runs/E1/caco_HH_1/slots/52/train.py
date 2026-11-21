"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
        m = Chem.MolFromSmiles(smi)
        fp = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6).astype(np.float32)
        _CACHE[smi] = np.concatenate([fp, desc])
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


def phys_one(smi):
    m = Chem.MolFromSmiles(smi)
    n = max(m.GetNumHeavyAtoms(), 1)
    arom = sum(a.GetIsAromatic() for a in m.GetAtoms()) / n
    vals = [f(m) for f in (Descriptors.TPSA, Descriptors.MolLogP, Descriptors.NumHDonors,
                           Descriptors.NumHAcceptors, Descriptors.MolWt, Descriptors.NumRotatableBonds)]
    return vals + [arom]


def phys(smiles):
    return np.array([phys_one(s) for s in smiles], dtype=np.float64)


LOCAL_K = 60
LOCAL_RIDGE = 1.0


def local_linear(Z, y, Q, loo=False):
    # Z: standardized training physchem, Q: query rows; loo=True excludes each training point from its own neighbours
    D = np.sqrt(((Q[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    if loo:
        np.fill_diagonal(D, np.inf)
    out = np.empty(len(Q))
    for i in range(len(Q)):
        idx = np.argsort(D[i])[:LOCAL_K]
        d = D[i, idx]
        h = max(np.median(d), 1e-6)
        w = np.exp(-0.5 * (d / h) ** 2)
        A = np.hstack([np.ones((len(idx), 1)), Z[idx]])
        AtW = A.T * w
        R = LOCAL_RIDGE * np.eye(A.shape[1])
        R[0, 0] = 0.0
        beta = np.linalg.solve(AtW @ A + R, AtW @ y[idx])
        out[i] = beta[0] + Q[i] @ beta[1:]
    return out


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    scaler = StandardScaler().fit(ptr)
    Ztr = scaler.transform(ptr)
    Zte = scaler.transform(pte)
    base_tr = local_linear(Ztr, y, Ztr, loo=True)
    base_te = local_linear(Ztr, y, Zte)
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    params = dict(tree_method='hist', max_depth=6, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                  random_state=seed, n_jobs=8)
    stage1 = XGBRegressor(objective='reg:squarederror', n_estimators=200, **params)
    stage1.fit(Xtr, y - base_tr)
    stage2 = XGBRegressor(objective='reg:absoluteerror', n_estimators=400, **params)
    stage2.fit(Xtr, y - base_tr, xgb_model=stage1.get_booster())
    return base_te + stage2.predict(Xte)
