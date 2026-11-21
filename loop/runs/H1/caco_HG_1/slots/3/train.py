"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from rdkit.Chem import Crippen
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1

def get_physchem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return np.zeros(10)
    # [MolLogP, TPSA, MW, HBD, HBA, RotBonds, AromaticRings, FormalCharge, Fsp3, AcidicGroups]
    return np.array([ 
        Crippen.MolLogP(mol), 
        Descriptors.TPSA(mol), 
        Descriptors.MolWt(mol), 
        Descriptors.NumHDonors(mol), 
        Descriptors.NumHAcceptors(mol), 
        Descriptors.NumRotatableBonds(mol), 
        Descriptors.NumAromaticRings(mol), 
        Descriptors.MolCharge(mol), 
        Descriptors.FractionCSP3(mol), 
        sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8 and atom.GetHybridization() == Chem.rdkit.Chem.GetHybridizationState.SP3) 
    ], dtype=np.float32)

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
    phys = np.array([get_physchem(s) for s in smiles])
    return np.hstack([fps, phys])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
