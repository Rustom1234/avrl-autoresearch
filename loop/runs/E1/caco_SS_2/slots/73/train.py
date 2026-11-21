"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from lightgbm import LGBMRegressor, early_stopping
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append(np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def extra_feats(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=512, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
        fc = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        rows.append(np.concatenate([maccs, fc]))
    return np.array(rows)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    def make(n):
        return LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    Atr = np.hstack([Xtr, extra_feats(train_smiles)])
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in train_smiles])
    uniq = np.unique(scaf)
    rng = np.random.RandomState(seed)
    vg = rng.choice(uniq, max(1, int(0.15 * len(uniq))), replace=False)
    vm = np.isin(scaf, vg)
    es = make(1500)
    es.fit(Atr[~vm], y[~vm], eval_set=[(Atr[vm], y[vm])], eval_metric='l1',
           callbacks=[early_stopping(50, first_metric_only=True, verbose=False)])
    best = es.best_iteration_ or 500
    model = make(max(50, int(best * 1.1)))
    model.fit(Atr, y)
    p_lgb = model.predict(np.hstack([Xte, extra_feats(test_smiles)]))
    keep = Xtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
    Str = qt.fit_transform(Xtr[:, keep])
    Ste = qt.transform(Xte[:, keep])
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Ste)
