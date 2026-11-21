"""train.py: predicts a measured property of small molecules from their SMILES strings (regression).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        fp = gen.GetFingerprintAsNumPy(mol)
        desc = np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                          Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), 
                          Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)])
        feats.append(np.concatenate([fp, desc]))
    return np.array(feats, dtype=np.float32)


from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

def fit_predict(train_smiles, train_y, test_smiles, seed):
    X_train = featurize(train_smiles)
    y_train = np.asarray(train_y, dtype=np.float64)
    eps = 1e-6
    y_train_log = np.log10(y_train + eps)
    
    # Generate Murcko scaffolds for GroupKFold
    scaffolds = []
    for s in train_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol: scaffolds.append(mol.RemoveHs().Getius(Chem.rdkit.Chem.rdChem.Core.ScaffoldHelpers.GetScaffold_2D(mol))) # Logic error fix: simplified
    # Re-simplifying for reliability in script:
    scaffolds = [Chem.rdkit.Chem.rdChem.Core.ScaffoldHelpers.GetScaffold_2D(Chem.MolFromSmiles(s)) for s in train_smiles]
    
    # Actually, use standard RDKit scaffold extraction
    from rdkit import Chem
    from rdkit.Chem import rdchem
    scaffolds = [rdchem.Scaffold_2D(Chem.MolFromSmiles(s))[0] for s in train_smiles]
    
    # Better way to handle scaffolds for GroupKFold
    scaffolds = []
    for s in train_smiles:
        m = Chem.MolFromSmiles(s)
        if m: scaffolds.append(m.GetSubstructMatches(rdchem.rdchem.rdchem.Core.ScaffoldHelpers.GetScaffold_2D(m))) 
    # Wait, let's just use the SMILES as unique identifiers or simplified scaffolds
    # To stay safe and simple, we use the scaffold as a string to group
    scaffold_keys = []
    for s in train_smiles:
        m = Chem.MolFromSmiles(s)
        if m: scaffold_keys.append(m.GetScaffold_2D().ToSmiles())
        else: scaffold_keys.append("unknown")
    
    gkf = GroupKFold(n_splits=3)
    
    best_score = float('inf')
    best_params = None
    
    # Hyperparameter search space
    depths = [3, 4, 5, 6, 7]
    lrs = [0.02, 0.03, 0.05]
    subs = [0.6, 0.8, 1.0]
    col_samples = [0.3, 0.5, 0.7]
    min_weights = [1, 3, 5]
    
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(6):
        configs.append({
            'max_depth': rng.choice(depths),
            'learning_rate': rng.choice(lrs),
            'subsample': rng.choice(subs),
            'colsample_bytree': rng.choice(col_samples),
            'min_child_weight': rng.choice(min_weights)
        })
    
    for params in configs:
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train_log, groups=scaffold_keys):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
            
            m = XGBRegressor(**params, n_estimators=300, random_state=seed, 
                               n_jobs=-1, tree_method='hist')
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            corr, _ = spearmanr(y_val, preds)
            fold_scores.append(1 - corr)
        
        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            
    final_model = XGBRegressor(**best_params, n_estimators=500, random_state=seed, 
                                 n_jobs=-1, tree_method='hist')
    final_model.fit(X_train, y_train_log)
    return final_model.predict(featurize(test_smiles))
