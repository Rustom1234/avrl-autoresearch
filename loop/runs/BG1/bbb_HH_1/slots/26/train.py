"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

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
    train_smi = list(train_smiles)
    test_smi = list(test_smiles)
    vec = CountVectorizer(analyzer="char", ngram_range=(2, 6), min_df=5, lowercase=False, max_features=1000)
    X_train = sparse.hstack([sparse.csr_matrix(featurize(train_smi)), vec.fit_transform(train_smi)]).tocsr()
    X_test = sparse.hstack([sparse.csr_matrix(featurize(test_smi)), vec.transform(test_smi)]).tocsr()
    model.fit(X_train, np.asarray(train_y, dtype=np.int64))
    return model.predict_proba(X_test)[:, 1]
