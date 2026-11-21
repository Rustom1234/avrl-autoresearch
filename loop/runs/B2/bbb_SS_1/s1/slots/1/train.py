"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
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
    # Inner scaffold-grouped split to choose boosting iteration counts.
    groups = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles]
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups=groups))
    lgb_kw = dict(learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.3, n_jobs=8, random_state=seed, verbose=-1)
    lgb_probe = LGBMClassifier(n_estimators=1500, **lgb_kw)
    lgb_probe.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric='auc',
                  callbacks=[early_stopping(100, verbose=False)])
    n_lgb = lgb_probe.best_iteration_ or 1500
    cb_kw = dict(learning_rate=0.03, depth=6, rsm=0.2, thread_count=8, random_seed=seed, verbose=0)
    cb_probe = CatBoostClassifier(iterations=1500, eval_metric='AUC', od_type='Iter', od_wait=100,
                                  use_best_model=True, **cb_kw)
    cb_probe.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
    n_cb = (cb_probe.get_best_iteration() or 0) + 1 or 1500
    models = [
        RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed),
        LGBMClassifier(n_estimators=n_lgb, **lgb_kw),
        CatBoostClassifier(iterations=n_cb, **cb_kw),
    ]
    preds = []
    for m in models:
        m.fit(X, y)
        preds.append(rankdata(m.predict_proba(Xt)[:, 1]) / len(Xt))
    return np.mean(preds, axis=0)
