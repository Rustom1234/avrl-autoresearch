"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


N_DESC = 100


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def descriptors(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        r = []
        for _, f in Descriptors.descList:
            try:
                r.append(f(m))
            except Exception:
                r.append(0.0)
        rows.append(r)
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def erg(smiles):
    from rdkit.Chem import rdReducedGraphs
    return np.array([rdReducedGraphs.GetErGFingerprint(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def minmax_kernel(A, B):
    sa, sb = A.sum(1), B.sum(1)
    K = np.empty((len(A), len(B)), dtype=np.float64)
    for i in range(len(A)):
        mn = np.minimum(A[i], B).sum(1)
        K[i] = mn / np.maximum(sa[i] + sb - mn, 1e-9)
    return K


def fit_predict(train_smiles, train_y, test_smiles, seed):
    from scipy.stats import rankdata
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    y = np.asarray(train_y, dtype=np.float64)
    y = rankdata(y) / len(y)
    Xtr = np.hstack([descriptors(train_smiles), erg(train_smiles), featurize(train_smiles)])
    Xte = np.hstack([descriptors(test_smiles), erg(test_smiles), featurize(test_smiles)])
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from lightgbm import early_stopping
    # scaffold-grouped 15% inner holdout for early stopping of the boosters
    rng = np.random.RandomState(seed)
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in train_smiles])
    uniq = np.unique(scaf)
    rng.shuffle(uniq)
    hold_set = set(uniq[:max(1, int(round(0.15 * len(uniq))))])
    va = np.array([s in hold_set for s in scaf])
    tr = ~va
    preds = []
    rf = RandomForestRegressor(n_estimators=300, max_features=0.2, n_jobs=8, random_state=seed).fit(Xtr, y)
    preds.append(rankdata(rf.predict(Xte)) / len(Xte))
    lgb_kw = dict(learning_rate=0.03, num_leaves=15, colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                  min_child_samples=10, n_jobs=8, random_state=seed, verbose=-1)
    lg = LGBMRegressor(n_estimators=1000, **lgb_kw)
    lg.fit(Xtr[tr], y[tr], eval_set=[(Xtr[va], y[va])], callbacks=[early_stopping(50, verbose=False)])
    n_lgb = int(round(1.1 * (lg.best_iteration_ or 1000)))
    lg = LGBMRegressor(n_estimators=n_lgb, **lgb_kw).fit(Xtr, y)
    preds.append(rankdata(lg.predict(Xte)) / len(Xte))
    cb = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           random_seed=seed, verbose=0, early_stopping_rounds=50)
    cb.fit(Xtr[tr], y[tr], eval_set=(Xtr[va], y[va]))
    n_cb = int(round(1.1 * (cb.get_best_iteration() + 1)))
    cb = CatBoostRegressor(iterations=n_cb, learning_rate=0.05, depth=6, rsm=0.2, thread_count=8,
                           random_seed=seed, verbose=0)
    cb.fit(Xtr, y)
    preds.append(rankdata(cb.predict(Xte)) / len(Xte))
    from sklearn.linear_model import Ridge
    r = Ridge(alpha=10.0).fit(np.log1p(Xtr[:, -N_BITS:]), y)
    pr = r.predict(np.log1p(Xte[:, -N_BITS:]))
    return 0.7 * np.mean(preds, axis=0) + 0.3 * rankdata(pr) / len(pr)
