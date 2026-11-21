"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

# Ionization SMARTS
SMARTS_PATTERNS = [
     Chem.MolFromSmarts('[OW]C(=O)O'),              # Carboxylic acids
     Chem.MolFromSmarts('c1n[nH]n[nH]1', 1),        # Tetrazoles
     Chem.MolFromSmarts('S(=O)(=O)NC(=O)C', 1),      # Acyl sulfonamides
     Chem.MolFromSmarts('[NH2]C', 1),                # Primary aliphatic amines
     Chem.MolFromSmarts('[NH]C', 1),                 # Secondary aliphatic amines
     Chem.MolFromSmarts('[N]C', 1),                  # Tertiary aliphatic amines
     Chem.MolFromSmarts('C=[NH]+', 1),               # Amidines
     Chem.MolFromSmarts('N=C=N', 1),                 # Guanidines
     Chem.MolFromSmarts('c1ccc(=N)cc1', 1),          # Pyridine-like nitrogens
     Chem.MolFromSmarts('N1[+]', 1)                  # Quaternary nitrogens
]
# Filter out None patterns from failed SMARTS
SMARTS_PATTERNS = [p for p in SMARTS_PATTERNS if p is not None]

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)])
        
        ion_counts = np.array([len(mol.GetSubstructMatches(p)) for p in SMARTS_PATTERNS], dtype=np.float32)
        feats.append(np.concatenate([fp, desc, ion_counts]))
    return np.array(feats, dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03, 
                           subsample=0.8, colsample_bytree=0.5, 
                           min_child_weight=3, random_state=seed, 
                           n_jobs=-1, tree_method='hist')
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    eps = 1e-6
    y_train_log = np.log10(y_train + eps)
    model.fit(X_train, y_train_log)
    return model.predict(featurize(test_smiles))
