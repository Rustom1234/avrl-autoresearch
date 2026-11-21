"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05


def ring_feats(m):
    ri = m.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    # fused ring systems: merge rings sharing at least two atoms
    systems = []
    for r in rings:
        new = set(r)
        keep = []
        for s in systems:
            if len(s & new) >= 2:
                new |= s
            else:
                keep.append(s)
        systems = keep + [new]
    n_sys = len(systems)
    largest = max((len(s) for s in systems), default=0)
    shared = sum(1 for b in m.GetBonds() if ri.NumBondRings(b.GetIdx()) > 1)
    sys_of = {}
    for k, s in enumerate(systems):
        for a in s:
            sys_of[a] = k
    # longest acyclic non-ring linker touching at least two ring systems
    seen = set()
    linker = 0
    for a in range(m.GetNumAtoms()):
        if a in sys_of or a in seen:
            continue
        comp, stack, touch = [a], [a], set()
        seen.add(a)
        while stack:
            x = stack.pop()
            for nb in m.GetAtomWithIdx(x).GetNeighbors():
                j = nb.GetIdx()
                if j in sys_of:
                    touch.add(sys_of[j])
                elif j not in seen:
                    seen.add(j)
                    comp.append(j)
                    stack.append(j)
        if len(touch) >= 2:
            linker = max(linker, len(comp))
    nh = max(1, m.GetNumHeavyAtoms())
    ring_frac = len(sys_of) / nh
    rot_ratio = Descriptors.NumRotatableBonds(m) / nh
    multi = sum(1 for a in range(m.GetNumAtoms()) if ri.NumAtomRings(a) > 1)
    return [n_sys, largest, shared, linker, ring_frac, rot_ratio, len(rings), multi]


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = np.zeros(N_BITS, dtype=np.float32)
        for k, v in AllChem.GetHashedMorganFingerprint(m, RADIUS, nBits=N_BITS).GetNonzeroElements().items():
            fp[k] = v
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float32)
        rows.append(np.concatenate([fp, desc, np.array(ring_feats(m), dtype=np.float32)]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def base_features(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([Descriptors.TPSA(m), Descriptors.MolWt(m), Descriptors.NumHDonors(m), Descriptors.MolLogP(m)])
    return np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    y = np.asarray(train_y, dtype=np.float64)
    A_tr = np.c_[np.ones(len(y)), base_features(train_smiles)]
    A_te = np.c_[np.ones(len(test_smiles)), base_features(test_smiles)]
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), y - A_tr @ coef)
    return model.predict(featurize(test_smiles)) + A_te @ coef
