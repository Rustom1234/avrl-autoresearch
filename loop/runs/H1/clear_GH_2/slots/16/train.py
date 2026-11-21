"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from lightgbm import LGBMRegressor

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
        rows.append(np.concatenate([fp, desc]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(num_leaves=31, learning_rate=0.05, colsample_bytree=0.8, n_estimators=N_TREES,
                          n_jobs=8, random_state=seed, verbose=-1)
    model.fit(featurize(train_smiles), np.log1p(np.asarray(train_y, dtype=np.float64)))
    return np.expm1(model.predict(featurize(test_smiles)))
