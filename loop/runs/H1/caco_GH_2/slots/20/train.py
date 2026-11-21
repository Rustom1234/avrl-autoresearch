"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdFingerprintGenerator, rdMolDescriptors
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
        rows.append(np.append(gen.GetFingerprintAsNumPy(mol), extra))
    return np.array(rows, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    y = np.asarray(train_y, dtype=np.float64)
    edges = np.histogram_bin_edges(y, bins=10)
    idx = np.clip(np.digitize(y, edges[1:-1]), 0, 9)
    w = 1.0 / np.bincount(idx, minlength=10)[idx]
    model.fit(featurize(train_smiles), y, sample_weight=w)
    return model.predict(featurize(test_smiles))
