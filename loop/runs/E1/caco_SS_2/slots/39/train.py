"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from rdkit.Chem import Crippen, rdMolDescriptors, Lipinski
from sklearn.preprocessing import QuantileTransformer

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        charges = [a.GetFormalCharge() for a in m.GetAtoms()]
        rows.append(np.array([
            Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m), Descriptors.MolWt(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m), Lipinski.NumRotatableBonds(m),
            rdMolDescriptors.CalcFractionCSP3(m), rdMolDescriptors.CalcNumAromaticRings(m),
            sum(abs(c) for c in charges), sum(c != 0 for c in charges), sum(charges),
            rdMolDescriptors.CalcLabuteASA(m)], dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    model = GradientBoostingRegressor(loss='huber', n_estimators=400, learning_rate=0.03, max_depth=3,
                                      subsample=0.8, min_samples_leaf=5, random_state=seed)
    model.fit(Xtr, y)
    return model.predict(Xte)
