"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1
# Ionizable-group SMARTS patterns (logD at pH 7.4 depends strongly on ionization)
IONIZABLE = [Chem.MolFromSmarts(p) for p in [
    '[CX3](=O)[OX2H1,OX1-]',                       # carboxylic acid
    '[#6]1:[#7]:[#7]:[#7]:[#7]:1',                 # tetrazole
    'c[OX2H1]',                                    # phenol
    '[NX3;H2;!$(NC=[O,S,N]);!$(N-a)][CX4]',        # primary aliphatic amine
    '[NX3;H1;!$(NC=[O,S,N]);!$(N-a)]([CX4])[CX4]', # secondary aliphatic amine
    '[NX3;H0;!$(NC=[O,S,N]);!$(N-a)]([CX4])([CX4])[CX4]',  # tertiary aliphatic amine
    '[NX3;!$(NC=O)]-[CX3;!$(C=O)]=[NX2;!$(N-[O,S])]',      # amidine / guanidine
    '[nX2;r6]',                                    # pyridine-type basic N
    'S(=O)(=O)[OX2H1,OX1-]',                       # sulfonic acid
    '[NX4+]',                                      # quaternary nitrogen
]]


def featurize(smiles):
    # Morgan counts, atom-pair counts and topological-torsion counts, followed by 167 MACCS keys
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS,
                                                    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024)
    tt_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024)
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mk = np.zeros((167,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), mk)
        ion = np.array([len(mol.GetSubstructMatches(q)) for q in IONIZABLE], dtype=np.float32)
        mk = np.concatenate([mk, ion])
        rows.append(np.concatenate([gen.GetCountFingerprintAsNumPy(mol).astype(np.float32),
                                    ap_gen.GetCountFingerprintAsNumPy(mol).astype(np.float32),
                                    tt_gen.GetCountFingerprintAsNumPy(mol).astype(np.float32), mk]))
    return np.array(rows, dtype=np.float32)


_DESC_CACHE = {}


def describe(smiles):
    # RDKit physchem descriptors, cached per SMILES across folds
    if smiles not in _DESC_CACHE:
        mol = Chem.MolFromSmiles(smiles)
        _DESC_CACHE[smiles] = np.array([f(mol) for _, f in Descriptors._descList], dtype=np.float64)
    return _DESC_CACHE[smiles]


def fit_lgb(X, y, X_te, seed):
    # Inner 10% validation split (stratified by target quantile) to pick the tree count, then refit on all data
    bins = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    idx_tr, idx_va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=bins, random_state=seed)
    params = dict(objective='l1', learning_rate=0.05, random_state=seed, n_jobs=8, verbose=-1)
    m = lgb.LGBMRegressor(n_estimators=3000, **params)
    m.fit(X[idx_tr], y[idx_tr], eval_set=[(X[idx_va], y[idx_va])], eval_metric='l1',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    best = m.best_iteration_ if m.best_iteration_ > 0 else 3000
    full = lgb.LGBMRegressor(n_estimators=max(1, int(best * 1.1)), **params)
    full.fit(X, y)
    return full.predict(X_te)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    model = RandomForestRegressor(n_estimators=N_TREES, max_features=MAX_FEATURES, min_samples_leaf=MIN_SAMPLES_LEAF,
                                  n_jobs=-1, random_state=seed)
    model.fit(featurize(train_smiles), y)
    p_rf = model.predict(featurize(test_smiles))

    D_tr = np.nan_to_num(np.vstack([describe(s) for s in train_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    D_te = np.nan_to_num(np.vstack([describe(s) for s in test_smiles]), nan=0.0, posinf=0.0, neginf=0.0)
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=seed)
    Z_tr = qt.fit_transform(D_tr)
    Z_te = qt.transform(D_te)
    svr = SVR(C=1.0, epsilon=0.1, gamma='scale')
    svr.fit(Z_tr, y)
    p_svr = svr.predict(Z_te)
    p_lgb = fit_lgb(featurize(train_smiles), y, featurize(test_smiles), seed)
    F_tr, F_te = featurize(train_smiles)[:, :N_BITS], featurize(test_smiles)[:, :N_BITS]
    cb = CatBoostRegressor(loss_function='MAE', iterations=1000, learning_rate=0.05, depth=6, random_seed=seed,
                           verbose=0, thread_count=8)
    cb.fit(np.hstack([F_tr, D_tr]), y)
    p_cb = cb.predict(np.hstack([F_te, D_te]))
    return (p_rf + p_svr + p_lgb + p_cb) / 4.0
