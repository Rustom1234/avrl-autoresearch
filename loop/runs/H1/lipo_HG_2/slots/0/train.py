"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


IONIZABLE_SMARTS = [
    r'[CX3](=O)[OX2H1,OX1-]',  # Carboxylic acids
    r'[NX3;H2,H1,H0;!$(NC=O);!$(N-a);!$(N-S(=O))]',  # Aliphatic amines
    r'N=[N+]=N',  # Amidines
    r'c1ncnnc1'  # Tetrazoles
]

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    
    extra = []
    for m in mols:
        counts = []
        for smarts in IONIZABLE_SMARTS:
            p = Chem.recover_pattern_from_smarts(smarts)
            counts.append(float(m.GetSubstructMatches(p)))
        extra.append(counts)
    
    return np.hstack([fps, np.array(extra, dtype=np.float32)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
