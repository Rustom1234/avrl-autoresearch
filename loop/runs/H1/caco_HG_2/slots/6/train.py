"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_ESTIMATORS = 700
LEARNING_RATE = 0.03

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS, 
        fpSize=N_BITS, 
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    )
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Morgan fingerprints
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    # RDKit descriptors
    desc_list = Descriptors.descList
    desc_data = []
    for m in mols:
        d = [float(Descriptors.Calc(desc_list[i])) for i in range(len(desc_list))]
        d = [0.0 if not np.isfinite(x) else x for x in d]
        d = [min(max(x, -1e6), 1e6) for x in d]
        desc_data.append(d)
    desc = np.array(desc_data, dtype=np.float32)
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, 
                           max_depth=6, subsample=0.8, colsample_bytree=0.5, 
                           objective='reg:absoluteerror', n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
