"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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


def laplace_boost(Xtr, r, Xte, seed, n_rounds=40, step=0.1):
    # Laplace(mu, b) residual model; natural-gradient steps (Fisher: 1/b^2 for mu, 1 for s)
    dtr = xgb.DMatrix(Xtr)
    dte = xgb.DMatrix(Xte)
    p = dict(max_depth=4, eta=0.1, tree_method='hist', seed=seed, nthread=8,
             objective='reg:squarederror')
    mu0 = float(np.median(r))
    s0 = float(np.log(max(np.mean(np.abs(r - mu0)), 1e-3)))
    mu_tr, mu_te = np.full(len(r), mu0), np.full(Xte.shape[0], mu0)
    s_tr, s_te = np.full(len(r), s0), np.full(Xte.shape[0], s0)
    for _ in range(n_rounds):
        x = r - mu_tr
        b = np.exp(s_tr)
        # descent directions = negative natural gradients of the Laplace NLL
        d_mu = b * np.sign(x)
        d_s = np.abs(x) / b - 1.0
        dtr.set_label(d_mu)
        bst = xgb.train(p, dtr, num_boost_round=60)
        mu_tr += step * bst.predict(dtr)
        mu_te += step * bst.predict(dte)
        dtr.set_label(d_s)
        bst = xgb.train(p, dtr, num_boost_round=60)
        s_tr += step * bst.predict(dtr)
        s_te += step * bst.predict(dte)
    return mu_te


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    ptr = phys(train_smiles)
    pte = phys(test_smiles)
    scaler = StandardScaler().fit(ptr)
    lin = Ridge(alpha=1.0).fit(scaler.transform(ptr), y)
    base_tr = lin.predict(scaler.transform(ptr))
    base_te = lin.predict(scaler.transform(pte))
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    return base_te + laplace_boost(Xtr, y - base_tr, Xte, seed)
