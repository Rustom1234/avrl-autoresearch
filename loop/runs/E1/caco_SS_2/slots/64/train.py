"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, MACCSkeys
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import QuantileTransformer
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_PATTS = [Chem.MolFromSmarts(p) for p in (
    'C(=O)[OX2H1,OX1-]',
    '[NX3;H2,H1,H0;!$(NC=[O,S,N]);!$(N-a);!$(N-S(=O)=O)]',
    'C(=O)[NX3]')]

# Settings
N_BITS = 1024
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        m = _CHOOSER.choose(m)
        d = [f(m) for _, f in Descriptors.descList]
        mw = Descriptors.MolWt(m)
        nb = max(1, m.GetNumBonds())
        d += [Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
              rdMolDescriptors.CalcTPSA(m) / max(mw, 1.0),
              rdMolDescriptors.CalcNumRotatableBonds(m) / nb]
        d += [len(m.GetSubstructMatches(p)) for p in _PATTS]
        rows.append(np.array(d, dtype=np.float64))
    X = np.array(rows)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def extra_feats(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=512, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32)
        fc = gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
        rows.append(np.concatenate([maccs, fc]))
    return np.array(rows)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, min_child_samples=10, verbose=-1, random_state=seed, n_jobs=8)
    Xtr, Xte = featurize(train_smiles), featurize(test_smiles)
    y = np.asarray(train_y, dtype=np.float64)
    model.fit(np.hstack([Xtr, extra_feats(train_smiles)]), y)
    p_lgb = model.predict(np.hstack([Xte, extra_feats(test_smiles)]))
    keep = Xtr.std(axis=0) > 0
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(200, len(Xtr)), random_state=seed)
    Str = qt.fit_transform(Xtr[:, keep])
    Ste = qt.transform(Xte[:, keep])
    svr = SVR(kernel='rbf', C=3, epsilon=0.1, gamma='scale').fit(Str, y)
    return 0.5 * p_lgb + 0.5 * svr.predict(Ste)
