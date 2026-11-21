"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import rankdata
import xgboost as xgb
from sklearn.metrics import log_loss
from rdkit.Chem.Scaffolds import MurckoScaffold

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
BASIC_AMINE = Chem.MolFromSmarts('[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S(=O)=O)]')


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([
            rdMolDescriptors.CalcTPSA(m),
            Descriptors.MolWt(m),
            Descriptors.MolLogP(m),
            rdMolDescriptors.CalcNumHBD(m),
            rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcFractionCSP3(m),
            Chem.GetFormalCharge(m),
            len(m.GetSubstructMatches(ACID)),
            len(m.GetSubstructMatches(BASIC_AMINE)),
        ])
    return np.array(rows, dtype=np.float32)


def fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)
    x_morgan = np.array([morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_pair = np.array([pair.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    x_maccs = np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])
    return [x_morgan, x_pair, x_maccs]


BASE_PARAMS = {'objective': 'binary:logistic', 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.3,
               'min_child_weight': 2, 'nthread': 8}
CHUNK = 25
PATIENCE = 75
MAX_ROUNDS = 600
MIN_LR = 0.00625


def inner_holdout(smiles, seed):
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in smiles]
    rng = np.random.default_rng(seed)
    uniq = sorted(set(scaffolds))
    draw = dict(zip(uniq, rng.random(len(uniq))))
    return np.array([draw[s] < 0.2 for s in scaffolds])


def plateau_schedule(x_fit, y_fit, x_hold, y_hold, seed):
    d_fit = xgb.DMatrix(x_fit, label=y_fit)
    d_hold = xgb.DMatrix(x_hold)
    eta, etas, best, since, bst = 0.1, [], np.inf, 0, None
    while eta >= MIN_LR and CHUNK * len(etas) < MAX_ROUNDS:
        bst = xgb.train(dict(BASE_PARAMS, eta=eta, seed=seed), d_fit, CHUNK, xgb_model=bst)
        etas.append(eta)
        loss = log_loss(y_hold, bst.predict(d_hold))
        if loss < best - 1e-6:
            best, since = loss, 0
        else:
            since += CHUNK
        if since >= PATIENCE:
            eta, since = eta / 2, 0
    return etas


def replay_schedule(x, y, etas, seed):
    d = xgb.DMatrix(x, label=y)
    bst = None
    for eta in etas:
        bst = xgb.train(dict(BASE_PARAMS, eta=eta, seed=seed), d, CHUNK, xgb_model=bst)
    return bst


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    hold = inner_holdout(train_smiles, seed)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        etas = plateau_schedule(x_tr[~hold], y[~hold], x_tr[hold], y[hold], seed)
        bst = replay_schedule(x_tr, y, etas, seed)
        total += rankdata(bst.predict(xgb.DMatrix(x_te)))
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
