"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        # Count hydroxyls, amines, and carbonyls
        oh = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O' and atom.Get(*Chem.valence.IsAttachedToHydrogen(mol, atom)))
        # Simple heuristic for functional groups
        # Using Smarts patterns is safer but might be overkill; 
        # we'll stick to counts of specific atoms/patterns for simplicity
        nh = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N' and atom.Get(*Chem.valence.IsAttachedToHydrogen(mol, atom)))
        # Carbonyls (C=O)
        co = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O' and atom.GetBond0() and atom.GetBond0().GetBondType() == Chem.rdchem.BondType.DOUBLE and atom.GetBond0().GetAtom0().GetSymbol() == 'C')
        
        features.append(np.concatenate([fp, [oh, nh, co]], axis=0))
    return np.array(features, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
