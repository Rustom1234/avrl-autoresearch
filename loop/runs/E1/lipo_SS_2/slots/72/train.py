"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from rdkit.Chem import Crippen, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import lightgbm as lgb

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 500
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


GROUPS = [Chem.MolFromSmarts(p) for p in [
    "[CX3](=O)[OX1H0-,OX2H1]",
    "[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O);!$(N#*);!$(N=*)]([#6])",
    "[NX3][CX3]=[NX2]",
    "S(=O)(=O)[NX3]",
    "c[OX2H]",
    "c1nnn[nH]1",
    "[NX4+]",
    "[nX2;r6]",
]]


def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float64)
        desc = np.clip(np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0), -1e5, 1e5).astype(np.float32)
        grp = np.array([len(m.GetSubstructMatches(p)) for p in GROUPS], dtype=np.float32)
        rows.append(np.concatenate([fp, desc, grp]))
    return np.array(rows, dtype=np.float32)


def baseline_feats(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Crippen.MolLogP(m), Crippen.MolMR(m), rdMolDescriptors.CalcTPSA(m),
                     rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
                     len(m.GetSubstructMatches(GROUPS[0])), len(m.GetSubstructMatches(GROUPS[1]))])
    return np.array(rows, dtype=np.float64)


def tanimoto(A, B):
    a = A @ B.T
    return a / (A.sum(1)[:, None] + B.sum(1)[None, :] - a + 1e-9)


def base_preds(Xtr, ytr, Xte, seed, n_trees):
    et = ExtraTreesRegressor(n_estimators=n_trees, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                             n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    gb = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, min_child_samples=10, n_jobs=8, random_state=seed,
                           verbose=-1).fit(Xtr, ytr)
    Ftr, Fte = Xtr[:, :N_BITS], Xte[:, :N_BITS]
    sv = SVR(kernel="precomputed", C=5.0, epsilon=0.1).fit(tanimoto(Ftr, Ftr), ytr)
    return np.column_stack([et.predict(Xte), gb.predict(Xte), sv.predict(tanimoto(Fte, Ftr))])


def fit_predict(train_smiles, train_y, test_smiles, seed):
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    Btr, Bte = baseline_feats(train_smiles), baseline_feats(test_smiles)
    mu, sd = Btr.mean(0), Btr.std(0) + 1e-9
    rg = Ridge(alpha=1.0).fit((Btr - mu) / sd, np.asarray(train_y, dtype=np.float64))
    ctr, cte = rg.predict((Btr - mu) / sd), rg.predict((Bte - mu) / sd)
    y = np.asarray(train_y, dtype=np.float64) - ctr
    oof = np.zeros((len(y), 3))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(Xtr):
        oof[va] = base_preds(Xtr[tr], y[tr], Xtr[va], seed, 200)
    meta = LinearRegression(positive=True, fit_intercept=False).fit(oof, y)
    return meta.predict(base_preds(Xtr, y, Xte, seed, N_TREES)) + cte
