"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from scipy.stats import rankdata
from rdkit import Chem
import scipy.sparse as sp

# Settings
HIDDEN = 32
N_ROUNDS = 3
N_MODELS = 1
EPOCHS = 150
LR = 0.01
WEIGHT_DECAY = 1e-4
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]


def onehot(value, choices):
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1.0
    return out


def atom_features(atom):
    return (onehot(atom.GetSymbol(), ELEMENTS)
            + onehot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + onehot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            + onehot(atom.GetFormalCharge(), [-1, 0, 1])
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())])


def graphs(smiles):
    feats, rows, cols, seg = [], [], [], []
    offset = 0
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        feats.extend(atom_features(a) for a in mol.GetAtoms())
        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx() + offset, b.GetEndAtomIdx() + offset
            rows += [u, v]
            cols += [v, u]
        seg.extend([i] * mol.GetNumAtoms())
        offset += mol.GetNumAtoms()
    X = np.array(feats, dtype=np.float64)
    A = sp.csr_matrix((np.ones(len(rows)), (np.array(rows, dtype=int), np.array(cols, dtype=int))), shape=(offset, offset))
    P = sp.csr_matrix((np.ones(offset), (np.array(seg, dtype=int), np.arange(offset))), shape=(len(smiles), offset))
    return X, A, P


def init_params(rng, n_in):
    return {
        "Wi": rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_in, HIDDEN)),
        "bi": np.zeros(HIDDEN),
        "W1": rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), (HIDDEN, HIDDEN)),
        "W2": rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), (HIDDEN, HIDDEN)),
        "wo": rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), HIDDEN),
        "bo": np.zeros(1),
    }


def predict(p, X, A, P):
    h = np.tanh(X @ p["Wi"] + p["bi"])
    for _ in range(N_ROUNDS):
        h = np.tanh(h @ p["W1"] + (A @ h) @ p["W2"])
    return (P @ h) @ p["wo"] + p["bo"][0]


def gradients(p, X, A, P, y):
    # forward pass, keeping states for the hand-written backward pass
    hs = [np.tanh(X @ p["Wi"] + p["bi"])]
    ms = [None]
    for _ in range(N_ROUNDS):
        m = A @ hs[-1]
        ms.append(m)
        hs.append(np.tanh(hs[-1] @ p["W1"] + m @ p["W2"]))
    g = P @ hs[-1]
    pred = g @ p["wo"] + p["bo"][0]
    dpred = 2.0 * (pred - y) / len(y)
    grads = {"wo": g.T @ dpred, "bo": np.array([dpred.sum()])}
    dh = P.T @ (dpred[:, None] * p["wo"][None, :])
    grads["W1"] = np.zeros((HIDDEN, HIDDEN))
    grads["W2"] = np.zeros((HIDDEN, HIDDEN))
    for t in range(N_ROUNDS, 0, -1):
        dz = dh * (1.0 - hs[t] ** 2)
        grads["W1"] += hs[t - 1].T @ dz
        grads["W2"] += ms[t].T @ dz
        dh = dz @ p["W1"].T + A @ (dz @ p["W2"].T)
    dpre = dh * (1.0 - hs[0] ** 2)
    grads["Wi"] = X.T @ dpre
    grads["bi"] = dpre.sum(axis=0)
    return grads


def train_model(X, A, P, y, rng):
    p = init_params(rng, X.shape[1])
    mom = {k: np.zeros_like(v) for k, v in p.items()}
    vel = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(1, EPOCHS + 1):
        grads = gradients(p, X, A, P, y)
        for k in p:
            gk = grads[k] + WEIGHT_DECAY * p[k]
            mom[k] = b1 * mom[k] + (1 - b1) * gk
            vel[k] = b2 * vel[k] + (1 - b2) * gk ** 2
            p[k] -= LR * (mom[k] / (1 - b1 ** step)) / (np.sqrt(vel[k] / (1 - b2 ** step)) + eps)
    return p


def fit_predict(train_smiles, train_y, test_smiles, seed):
    X, A, P = graphs(train_smiles)
    Xt, At, Pt = graphs(test_smiles)
    y = (rankdata(np.asarray(train_y, dtype=np.float64)) - 0.5) / len(train_y)
    preds = []
    for k in range(N_MODELS):
        rng = np.random.default_rng([seed, k])
        p = train_model(X, A, P, y, rng)
        preds.append(predict(p, Xt, At, Pt))
    return np.mean(preds, axis=0)
