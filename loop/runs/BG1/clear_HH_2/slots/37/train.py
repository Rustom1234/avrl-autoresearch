"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import lightgbm as lgb
from rdkit.Chem import Descriptors
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

# Settings
N_BITS = 2048
RADIUS = 2
W_DESC = 0.6  # weight of the descriptor-based LightGBM rank; fingerprint ExtraTrees gets 1 - W_DESC


_DESC_CACHE = {}

SMARTS_PATTERNS = [Chem.MolFromSmarts(p) for p in (
    "[CX3](=O)[OX2H1,OX1-]",                                  # carboxylic acid
    "[#6]1:[#7]:[#7]:[#7]:[#7]:1",                            # tetrazole
    "[NX3;H2;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)][CX4]",          # primary aliphatic amine
    "[NX3;H1;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)]([CX4])[CX4]",   # secondary aliphatic amine
    "[NX3;H0;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)]([CX4])([CX4])[CX4]",  # tertiary aliphatic amine
    "[NX4+]",                                                 # quaternary N
    "[n]",                                                    # aromatic N-heterocycle
    "c[OX2H]",                                                # phenol
    "[CH3][O,N]",                                             # O-methyl or N-methyl
    "[CH2,CH3][c]",                                           # benzylic CH2/CH3
    "[#6][CX3](=O)[OX2][#6]",                                 # ester
    "[NX3][CX3](=O)",                                         # amide
    "[NX3]S(=O)(=O)",                                         # sulfonamide
    "c[F,Cl,Br,I]",                                           # aromatic halogen
)]


def desc_features(smiles):
    rows = []
    for s in smiles:
        if s not in _DESC_CACHE:
            m = Chem.MolFromSmiles(s)
            _DESC_CACHE[s] = [f(m) for _, f in Descriptors.descList] + [len(m.GetSubstructMatches(p)) for p in SMARTS_PATTERNS]
        rows.append(_DESC_CACHE[s])
    X = np.array(rows, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6)


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


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = rankdata(np.asarray(train_y, dtype=np.float64)) / len(train_y)

    lgbm = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, random_state=seed, n_jobs=8, verbose=-1)
    lgbm.fit(desc_features(train_smiles), y)
    p_desc = lgbm.predict(desc_features(test_smiles))

    X_tr, X_te = bit_fingerprints(train_smiles), bit_fingerprints(test_smiles)
    K_tr = tanimoto(X_tr, X_tr)
    s2 = choose_sigma2(K_tr, y, train_smiles)
    p_fp = gp_mean(K_tr, y, tanimoto(X_te, X_tr), s2)

    n = len(p_desc)
    return W_DESC * rankdata(p_desc) / n + (1 - W_DESC) * rankdata(p_fp) / n
