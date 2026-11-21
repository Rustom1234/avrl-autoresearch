"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from xgboost import XGBClassifier

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


N_PAIRS = 200000
L2_PAIR = 1e-3


def pairwise_linear_scores(train_smiles, y, test_smiles, bits_tr, bits_te, seed):
    # Curated descriptors are standardised; Morgan bits stay 0/1 so the pair-difference matrix remains sparse.
    d_tr = featurize(train_smiles).astype(np.float64)
    d_te = featurize(test_smiles).astype(np.float64)
    mu = d_tr.mean(axis=0)
    sd = d_tr.std(axis=0) + 1e-6
    x_tr = csr_matrix(np.hstack([(d_tr - mu) / sd, bits_tr.astype(np.float64)]))
    x_te = np.hstack([(d_te - mu) / sd, bits_te.astype(np.float64)])
    rng = np.random.default_rng(seed)
    p_idx = rng.choice(np.flatnonzero(y == 1), N_PAIRS)
    n_idx = rng.choice(np.flatnonzero(y == 0), N_PAIRS)
    diff = (x_tr[p_idx] - x_tr[n_idx]).tocsr()

    def obj(w):
        d = diff @ w
        loss = np.logaddexp(0, -d).mean() + L2_PAIR * (w @ w)
        grad = -(diff.T @ expit(-d)) / N_PAIRS + 2 * L2_PAIR * w
        return loss, grad

    res = minimize(obj, np.zeros(diff.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 100})
    return x_te @ res.x


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.int64)
    train_sets = fingerprints(train_smiles)
    test_sets = fingerprints(test_smiles)
    total = np.zeros(len(test_smiles))
    for x_tr, x_te in zip(train_sets, test_sets):
        model = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.3,
                              min_child_weight=2, eval_metric='logloss', random_state=seed, n_jobs=8)
        model.fit(x_tr, y)
        total += rankdata(model.predict_proba(x_te)[:, 1])
    total += rankdata(pairwise_linear_scores(train_smiles, y, test_smiles, train_sets[0], test_sets[0], seed))
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
