"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
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


def mlp_member(train_smiles, y, test_smiles, seed, n_nets=3, hidden=64, epochs=60, lr=1e-3, bs=64, wd=1e-4, clip=1.0):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)

    def build(smiles):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fp = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float64)
        return np.log1p(fp)

    x_tr = build(train_smiles)
    x_te = build(test_smiles)
    d_tr = featurize(train_smiles).astype(np.float64)
    d_te = featurize(test_smiles).astype(np.float64)
    mu = d_tr.mean(0)
    sd = d_tr.std(0) + 1e-6
    x_tr = np.hstack([x_tr, (d_tr - mu) / sd])
    x_te = np.hstack([x_te, (d_te - mu) / sd])
    yf = np.asarray(y, dtype=np.float64)
    n = len(yf)
    out = np.zeros(len(test_smiles))
    for k in range(n_nets):
        rng = np.random.default_rng(seed * 1009 + k)
        D = x_tr.shape[1]
        W1 = rng.normal(0, np.sqrt(2.0 / D), (D, hidden))
        b1 = np.zeros(hidden)
        W2 = rng.normal(0, np.sqrt(1.0 / hidden), (hidden, 1))
        b2 = np.zeros(1)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for _ in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = x_tr[idx]
                yb = yf[idx][:, None]
                h = np.maximum(xb @ W1 + b1, 0.0)
                p = 1.0 / (1.0 + np.exp(-(h @ W2 + b2)))
                dz = (p - yb) / len(idx)
                gW2 = h.T @ dz + wd * W2
                gb2 = dz.sum(0)
                dh = (dz @ W2.T) * (h > 0)
                gW1 = xb.T @ dh + wd * W1
                gb1 = dh.sum(0)
                grads = [gW1, gb1, gW2, gb2]
                norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
                if norm > clip:
                    grads = [g * (clip / norm) for g in grads]
                t += 1
                for j in range(len(params)):
                    g = grads[j]
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    mh = m[j] / (1 - 0.9 ** t)
                    vh = v[j] / (1 - 0.999 ** t)
                    params[j] -= lr * mh / (np.sqrt(vh) + 1e-8)
        h = np.maximum(x_te @ W1 + b1, 0.0)
        out += (1.0 / (1.0 + np.exp(-(h @ W2 + b2))))[:, 0]
    return out / n_nets


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
    total += rankdata(mlp_member(train_smiles, y, test_smiles, seed))
    total = (total - total.min()) / (total.max() - total.min() + 1e-12)
    return total
