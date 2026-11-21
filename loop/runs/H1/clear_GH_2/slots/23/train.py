"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, MACCSkeys
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 'sqrt'
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    props = rdMolDescriptors.Properties()
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        desc = np.array(props.ComputeProperties(mol), dtype=np.float32)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32)
        rows.append(np.concatenate([fp, desc, maccs]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y = np.log1p(np.asarray(train_y, dtype=np.float64))
    rf = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                               n_jobs=-1, random_state=seed)
    et = ExtraTreesRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                             n_jobs=-1, random_state=seed)
    rf.fit(X_train, y)
    et.fit(X_train, y)
    pred = 0.5 * (rf.predict(X_test) + et.predict(X_test))
    return np.expm1(pred)
