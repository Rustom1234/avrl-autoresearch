"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

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
    X = featurize(train_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=seed)
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=MAX_FEATURES, early_stopping_rounds=50, n_jobs=8, random_state=seed)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model.predict(featurize(test_smiles))
