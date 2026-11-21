"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor

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
        fp = gen.GetFingerprintAsNumPy(m)
        topo = [Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.MolLogP(m), rdMolDescriptors.CalcNumHBD(m),
                rdMolDescriptors.CalcNumHBA(m), rdMolDescriptors.CalcChi1n(m), rdMolDescriptors.CalcKappa2(m),
                Descriptors.BertzCT(m)]
        rows.append(np.concatenate([fp, np.nan_to_num(np.array(topo, dtype=np.float32))]))
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    def abs_corr(u, v):
        c = np.corrcoef(u, v)[0, 1]
        return 0.0 if not np.isfinite(c) else abs(c)

    X_train = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    X_test = featurize(test_smiles)
    # High-variance decorrelation: among the top-3 highest-variance features (training fold only),
    # drop the one less correlated with the target from any pair with |r| > 0.8.
    top = np.argsort(X_train.var(axis=0))[::-1][:3]
    keep = np.ones(X_train.shape[1], dtype=bool)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a, b = top[i], top[j]
        if keep[a] and keep[b] and abs_corr(X_train[:, a], X_train[:, b]) > 0.8:
            worse = a if abs_corr(X_train[:, a], y) < abs_corr(X_train[:, b], y) else b
            keep[worse] = False
    X_train, X_test = X_train[:, keep], X_test[:, keep]

    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(X_train, y)
    per_tree = np.stack([tree.predict(X_test) for tree in model.estimators_])
    return np.median(per_tree, axis=0)
