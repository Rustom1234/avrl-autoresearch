"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
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
        mol = Chem.MolFromSmiles(s)
        extra = [mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
                 Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
                 rdMolDescriptors.CalcNumRotatableBonds(mol)]
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float32)
        desc = np.array([fn(mol) for _, fn in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)
        rows.append(np.concatenate([gen.GetFingerprintAsNumPy(mol), np.array(extra), maccs, desc]))
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
