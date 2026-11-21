"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
import lightgbm as lgb

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
        desc = [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m)]
        rows.append(np.concatenate([fp, desc]))
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    n_val = max(1, int(0.1 * len(y)))
    n_tr = len(y) - n_val
    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.3, random_state=seed, n_jobs=8, verbose=-1)
    model.fit(X[:n_tr], y[:n_tr], eval_set=[(X[n_tr:], y[n_tr:])], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model.predict_proba(featurize(test_smiles))[:, 1]
