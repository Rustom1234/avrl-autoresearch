"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

# Settings
N_BITS = 2048
RADIUS = 2
W_DESC = 0.6  # weight of the descriptor-based LightGBM rank; fingerprint ExtraTrees gets 1 - W_DESC


_DESC_CACHE = {}


def desc_features(smiles):
    rows = []
    for s in smiles:
        if s not in _DESC_CACHE:
            m = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(m) for _, f in Descriptors.descList]
        rows.append(_DESC_CACHE[s])
    X = np.array(rows, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6)


def physchem(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m)])
    return np.array(rows, dtype=np.float64)


def count_features(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def bit_fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def tanimoto(A, B):
    inter = A @ B.T
    return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1e-9)


def gp_mean(K_tr, y, K_te, sigma2):
    mu = y.mean()
    alpha = np.linalg.solve(K_tr + sigma2 * np.eye(len(y)), y - mu)
    return mu + K_te @ alpha


def choose_sigma2(K, y, smiles):
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in smiles]
    groups = np.unique(scaffolds, return_inverse=True)[1]
    best_score, best_s2 = -np.inf, 0.3
    for s2 in (0.1, 0.3, 1.0):
        pred = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=5).split(K, y, groups):
            pred[va] = gp_mean(K[np.ix_(tr, tr)], y[tr], K[np.ix_(va, tr)], s2)
        score = spearmanr(pred, y).correlation
        if score > best_score:
            best_score, best_s2 = score, s2
    return best_s2


def gnn_features(smiles, seed, rounds=3):
    d = 11
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(0.0, 1.0, size=(d, d)) / np.sqrt(d) for _ in range(rounds)]
    bucket_of = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'Cl': 4, 'Br': 4, 'I': 4}
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        n = m.GetNumAtoms()
        h = np.zeros((n, d))
        A = np.zeros((n, n))
        for a in m.GetAtoms():
            i = a.GetIdx()
            h[i, bucket_of.get(a.GetSymbol(), 5)] = 1.0
            h[i, 6] = float(a.GetIsAromatic())
            h[i, 7] = a.GetDegree()
            h[i, 8] = a.GetTotalNumHs()
            h[i, 9] = float(a.GetFormalCharge() != 0)
            h[i, 10] = float(a.IsInRing())
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            A[i, j] = A[j, i] = 1.0
        deg = np.maximum(A.sum(1, keepdims=True), 1.0)
        pooled = []
        for W in Ws:
            h = np.tanh(h + (A @ (h @ W)) / deg)
            pooled.append(h.sum(0))
        out.append(np.concatenate(pooled))
    return np.array(out)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)
    A_tr = np.column_stack([np.ones(len(train_smiles)), physchem(train_smiles)])
    A_te = np.column_stack([np.ones(len(test_smiles)), physchem(test_smiles)])
    beta = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    base_tr, base_te = A_tr @ beta, A_te @ beta
    r = y - base_tr

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(train_smiles), r)
    p_desc = lgbm.predict(desc_features(test_smiles)) + base_te

    X_tr, X_te = bit_fingerprints(train_smiles), bit_fingerprints(test_smiles)
    K_tr = tanimoto(X_tr, X_tr)
    s2 = choose_sigma2(K_tr, r, train_smiles)
    p_fp = gp_mean(K_tr, r, tanimoto(X_te, X_tr), s2) + base_te

    G_tr, G_te = gnn_features(train_smiles, seed), gnn_features(test_smiles, seed)
    g_mu, g_sd = G_tr.mean(0), G_tr.std(0) + 1e-9
    ridge = RidgeCV(alphas=np.logspace(0, 3, 20)).fit((G_tr - g_mu) / g_sd, r)
    p_gnn = ridge.predict((G_te - g_mu) / g_sd) + base_te

    n = len(p_desc)
    return 0.8 * (W_DESC * rankdata(p_desc) / n + (1 - W_DESC) * rankdata(p_fp) / n) + 0.2 * rankdata(p_gnn) / n
