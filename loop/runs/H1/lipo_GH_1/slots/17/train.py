"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from lightgbm import LGBMRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 2000
LEARNING_RATE = 0.01
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.TPSA(mol), Descriptors.MolLogP(mol)])
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32)
        rows.append(np.concatenate([fp, desc, maccs]))
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=N_TREES, learning_rate=LEARNING_RATE, n_jobs=8, verbose=-1,
                          random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
