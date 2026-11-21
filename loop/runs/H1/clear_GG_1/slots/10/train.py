"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = AllChem.GetCountFingerprint(mol, N_BITS)
        arr = np.array(DataStructs.ConvertToNumpyArray(fp))
        features.append(np.append(arr, [mol.GetNumAtoms()]))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    y_train = np.log1p(np.asarray(train_y, dtype=np.float64))
    model.fit(featurize(train_smiles), y_train)
    preds = model.predict(featurize(test_smiles))
    return np.expm1(preds)
