"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVR

# Settings
N_BITS = 2048
RADIUS = 2
N_TREES = 300
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 1


ACID = Chem.MolFromSmarts('[CX3](=O)[OX2H1,OX1-]')
AMINE = Chem.MolFromSmarts('[NX3;!$(NC=[O,S,N]);!$(N-a);!$(NS=O)]')


def featurize(smiles):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([
            rdMolDescriptors.CalcTPSA(m), Crippen.MolLogP(m), Descriptors.MolWt(m),
            rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m),
            rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumAromaticRings(m),
            rdMolDescriptors.CalcFractionCSP3(m), Chem.GetFormalCharge(m),
            len(m.GetSubstructMatches(ACID)), len(m.GetSubstructMatches(AMINE)),
            m.GetNumHeavyAtoms()])
    return np.array(rows, dtype=np.float64)


def fit_predict(train_smiles, train_y, test_smiles, seed):
    qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=seed)
    Xtr = qt.fit_transform(featurize(train_smiles))
    model = SVR(kernel='rbf', C=3, gamma='scale', epsilon=0.1)
    model.fit(Xtr, np.asarray(train_y, dtype=np.float64))
    return model.predict(qt.transform(featurize(test_smiles)))
