"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from catboost import CatBoostRegressor

# Settings
N_BITS = 2048
RADIUS = 2
ITERS = 800
DEPTH = 6
LR = 0.05

ACID = Chem.MolFromSmarts('C(=O)[OX2H1,OX1-]')
BASIC_N = Chem.MolFromSmarts('[NX3;!$(N-C=[O,S,N]);!$(N-a);!$(N-S);!$(N=*);!$(N#*);!$(N-N);!$(N-O)]')
QUAT_N = Chem.MolFromSmarts('[N+,n+;!$([N+]-[O-]);!$([n+]-[O-]);!$([N+]=O)]')


def hand_feats(m):
    return [rdMolDescriptors.CalcTPSA(m), Descriptors.MolWt(m), Crippen.MolLogP(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m), Lipinski.NumRotatableBonds(m),
            rdMolDescriptors.CalcNumRings(m), rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m), m.GetNumHeavyAtoms(), Chem.GetFormalCharge(m),
            len(m.GetSubstructMatches(ACID)), len(m.GetSubstructMatches(BASIC_N)),
            len(m.GetSubstructMatches(QUAT_N))]


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        fp = np.zeros(N_BITS, dtype=np.float32)
        for k, v in AllChem.GetHashedMorganFingerprint(m, RADIUS, nBits=N_BITS).GetNonzeroElements().items():
            fp[k] = v
        desc = np.array([f(m) for _, f in Descriptors.descList], dtype=np.float32)
        hand = np.array(hand_feats(m), dtype=np.float32)
        rows.append(np.concatenate([fp, desc, hand]))
    return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = CatBoostRegressor(loss_function='MAE', depth=DEPTH, learning_rate=LR, iterations=ITERS,
                              random_seed=seed, verbose=0, thread_count=8)
    model.fit(featurize(train_smiles), np.asarray(train_y, dtype=np.float64))
    return model.predict(featurize(test_smiles))
