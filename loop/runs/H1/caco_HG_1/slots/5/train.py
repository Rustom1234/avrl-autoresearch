"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1

# Get the list of descriptors from RDKit
DESC_NAMES = Descriptors._descList

def featurize(smiles, desc_stats=None):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    features = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            features.append(np.zeros(N_BITS + len(DESC_NAMES), dtype=np.float32))
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.CalcMolDescriptor(mol, name) for name in DESC_NAMES], dtype=np.float32)
        # Replace NaN/Inf with 0
        desc[np.isnan(desc) | np.isinf(desc)] = 0
        features.append(np.concatenate([fp, desc]))
    
    feat_mat = np.array(features, dtype=np.float32)
    
    if desc_stats is not None:
        # desc_stats is a list of (low, high) tuples for the descriptor part
        # Note: the prompt asks for 1st and 99th percentiles computed on training fold
        # Since featurize is called twice, we apply the pre-computed limits here
        for i, (low, high) in enumerate(desc_stats):
            feat_mat[..., N_BITS + i] = np.clip(feat_mat[..., N_BITS + i], low, high)
            
    return feat_mat


def fit_predict(train_smiles, train_y, test_smiles, seed):
    x_train = featurize(train_smiles)
    
    # Calculate clipping bounds for descriptors
    desc_part = x_train[:, N_BITS:]
    desc_stats = []
    for i in range(desc_part.shape[1]):
        col = desc_part[:, i]
        desc_stats.append((np.percentile(col, 1), np.percentile(col, 99)))
    
    x_test = featurize(test_smiles, desc_stats=desc_stats)
    
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(x_train, np.asarray(train_y, dtype=np.float64))
    return model.predict(x_test)
