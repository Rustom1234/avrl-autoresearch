"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES_RF = 300
N_LEAVES_LGB = 31

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    X_test = featurize(test_smiles)
    y_train = np.asarray(train_y, dtype=np.int64)

    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=N_TREES_RF, max_features=0.3, min_samples_leaf=1, n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]

    # 2. LightGBM
    lgb_mod = lgb.LGBMClassifier(n_estimators=100, num_leaves=N_LEAVES_LGB, random_state=seed, verbose=-1)
    lgb_mod.fit(X_train, y_train)
    p_lgb = lgb_mod.predict_proba(X_test)[:, 1]

    # 3. Logistic Regression
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    lr = LogisticRegression(solver='lbfgs', random_state=seed)
    lr.fit(X_train_sc, y_train)
    p_lr = lr.predict_proba(X_test_sc)[:, 1]

    # 4. SVM
    svc = SVC(kernel='linear', probability=True, random_state=seed)
    svc.fit(X_train_sc, y_train)
    p_svc = svc.predict_proba(X_test_sc)[:, 1]

    # Rank averaging
    probs = np.array([p_rf, p_lgb, p_lr, p_svc])
    ranks = np.mean(np.argsort(np.sort(probs, axis=0), axis=1), axis=0)
    return ranks / (len(test_smiles) - 1)
