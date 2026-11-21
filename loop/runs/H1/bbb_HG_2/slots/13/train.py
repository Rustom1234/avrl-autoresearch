"""train.py: predicts a measured property of small molecules from their SMILES strings (binary classification; return the probability of class 1).

The harness calls fit_predict(train_smiles, train_y, test_smiles, seed) and scores the returned predictions.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.kernel_ridge import KernelRidge

# Settings
N_BITS = 2048
RADIUS = 2

def featurize(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    feats = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            # Use bit fingerprints for Tanimoto kernel
            fp = gen.GetFingerprintAsBitVect(mol)
            feats.append(np.array(fp)) # Store as bit vector for Tanimoto
    return np.array(feats, dtype=np.float32)

def get_tanimoto_kernel(fps):
    # fps are shape (N, 2048)
    # Tanimoto(a, b) = (a & b).count() / (a | b).count()
    # For binary vectors: (a * b).sum() / (a + b - a * b).sum()
    # Since they are bitsets, we can use a simplified distance
    # But for speed/simplicity, let's use the count of common bits
    # and total unique bits.
    # A standard approach for sklearn is to compute the matrix
    # Here we use a simple Dot product if normalized, but Tanimoto is specific.
    # Given the constraints, we'll use a similarity matrix.
    from rdkit import DataStructs
    fps_rdkit = [Chem.DataStructs.Taus_BitVect(f) for f in fps]
    # However, we can't easily iterate and call DataStructs.TanimotoSimilarity
    # for a full matrix without a loop. Let's use the dot product as a proxy
    # or a manual Tanimoto.
    # To be safe and fast:
    # Tanimoto(x, y) = sum(x * y) / (sum(x) + sum(y) - sum(x * y))
    # We use a small trick: convert to float and use matrix multiplication
    dot = np.dot(fps, fps.T)
    sums = np.sum(fps, axis=1).reshape(-1, 1) # (N, 1)
    denom = sums + sums.T - dot
    # Avoid division by zero
    return dot / np.maximum(denom, 1e-8)

def fit_predict(train_smiles, train_y, test_smiles, seed):
    train_fps = featurize(train_smiles)
    test_fps = featurize(test_smiles)
    
    # Kernel Ridge Regressor needs a square kernel for training
    K_train = get_tanimoto_kernel(train_fps)
    
    # Map y to {-1, 1}
    y_mapped = np.where(train_y == 1, 1.0, -1.0)
    
    # Fit
    model = KernelRidge(alpha=0.5, kernel='precomputed')
    model.fit(K_train, y_mapped)
    
    # Predict
    K_test = get_tanimoto_kernel(np.concatenate([train_fps, test_fps], axis=0))[
        train_fps.shape[0]:, :train_fps.shape[0]]
    # Wait, the kernel matrix must be (N_test, N_train)
    # Let's re-calculate K_test properly
    # K_test[i, j] = sim(test_i, train_j)
    # We need the similarity between test_fps and train_fps
    
    # Correcting logic: we need a kernel matrix of size (n_test, n_train)
    # Let's just use the full similarity matrix to get the right indices
    all_fps = np.concatenate([train_fps, test_fps], axis=0)
    K_all = get_tanimoto_kernel(all_fps)
    K_test_actual = K_all[train_fps.shape[0]:, :train_fps.shape[0]]
    
    out = model.predict(K_test_actual)
    # Convert back to probability via logistic squash
    return 1 / (1 + np.exp(-2 * out))
