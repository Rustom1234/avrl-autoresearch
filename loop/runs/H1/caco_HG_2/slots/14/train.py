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
    # RDKit descriptors and physicochemical rules
    desc_list = []
    for m in mols:
        # Basic descriptors
        mw = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        h_donors = Descriptors.NumHDonors(m)
        h_acceptors = Descriptors.NumHAcceptors(m)
        tpsa = Descriptors.TPSA(m)
        rot_bonds = Descriptors.NumRotBond(m)
        ar_rings = Descriptors.NumAromatic(m)
        formal_charge = Descriptors.FormalCharge(m)
        heavy_atoms = Descriptors.NumHeavyDos(m)
        
        # SMARTS matches for specific functional groups
        # Carboxylic acid and basic amines
        acids = len(Chem.MolFromSmarts('[OX1]C(=O)O').GetMatchesForMol(m))
        amines = len(Chem.MolFromSmarts('N[C;H2,H1,H0;?1]').GetMatchesForMol(m))
        
        desc_list.append([mw, logp, h_donors, h_acceptors, tpsa, rot_bonds, ar_rings, formal_charge, heavy_atoms, acids, amines])
    
    desc = np.array(desc_list, dtype=np.float32)
    return np.hstack([fps, desc])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=LEARNING_RATE, 
                           max_depth=6, subsample=0.8, colsample_bytree=0.5, 
                           objective='reg:absoluteerror', n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
