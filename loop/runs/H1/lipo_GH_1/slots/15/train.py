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
N_TREES = 500
LEARNING_RATE = 0.05
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


from rdkit.Chem import Descriptors


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        fp = gen.GetCountFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.TPSA(mol), Descriptors.MolLogP(mol)])
        rows.append(np.concatenate([fp, desc]))
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=N_TREES, learning_rate=LEARNING_RATE, n_jobs=8, verbose=-1,
                          random_state=seed, reg_alpha=0.1)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
