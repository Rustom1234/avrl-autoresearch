"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from scipy.stats import rankdata

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=np.float64)
    desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return np.hstack([fp, desc.astype(np.float32)])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.3, subsample=0.8, subsample_freq=1, min_child_samples=10,
                                  n_jobs=-1, random_state=seed)
    Xtr = featurize(train_smiles)
    Xte = featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.int64)
    model.fit(Xtr, y)
    p_lgbm = model.predict_proba(Xte)[:, 1]

    # Laplacian-corrected naive Bayes on binary ECFP4 bits
    btr = (Xtr[:, :N_BITS] > 0).astype(np.float64)
    bte = (Xte[:, :N_BITS] > 0).astype(np.float64)
    yf = y.astype(np.float64)
    prior = yf.mean()
    n_with = btr.sum(0)
    act_with = btr.T @ yf
    n_without = len(yf) - n_with
    act_without = yf.sum() - act_with
    w_with = np.log(((act_with + 1) / (n_with + 2)) / prior)
    w_without = np.log(((act_without + 1) / (n_without + 2)) / prior)
    nb_score = w_without.sum() + bte @ (w_with - w_without)
    p_nb = 1 / (1 + np.exp(-nb_score))

    # equal-weight rank average of the two predictors within the test set
    n = len(test_smiles)
    return (rankdata(p_lgbm) + rankdata(p_nb)) / (2 * n)
