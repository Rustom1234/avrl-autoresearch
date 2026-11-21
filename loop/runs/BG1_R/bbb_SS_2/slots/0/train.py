"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = [[f(Chem.MolFromSmiles(s)) for _, f in Descriptors.descList] for s in smiles]
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)
    Xtr = qt.fit_transform(featurize(train_smiles))
    Xte = qt.transform(featurize(test_smiles))
    model = SVC(C=1, gamma="scale", probability=True, random_state=seed)
    model.fit(Xtr, np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(Xte)[:, 1]
