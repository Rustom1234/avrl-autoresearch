"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


_CACHE = {}


def featurize_one(smi):
    if smi not in _CACHE:
        m = Chem.MolFromSmiles(smi)
        acid = Chem.MolFromSmarts('C(=O)[OH]')
        vals = [
            Descriptors.TPSA(m), Descriptors.MolLogP(m), Descriptors.MolWt(m),
            Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
            Descriptors.NumAromaticRings(m), Descriptors.RingCount(m), Descriptors.FractionCSP3(m),
            Descriptors.HeavyAtomCount(m),
            sum(a.GetFormalCharge() != 0 for a in m.GetAtoms()),
            len(m.GetSubstructMatches(acid)),
        ]
        _CACHE[smi] = np.nan_to_num(np.array(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return _CACHE[smi]


def featurize(smiles):
    return np.array([featurize_one(s) for s in smiles], dtype=np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES,
                                  min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed, n_jobs=8)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
