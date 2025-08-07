import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

K = 710
threshold = 1.0/8.0

id_mapping = {}

with open('id_mapping.csv', 'r') as f:
    for line in f:
        id1, idx = map(int, line.strip().split(' '))
        id_mapping[id1] = idx
        
# rev_mapping = {v: k for k, v in id_mapping.items()}

edges = []

sparse_matrix = sp.lil_matrix((len(id_mapping), len(id_mapping)))

with open('edges.csv', 'r') as f:
    for line in f:
        id1, id2 = map(int, line.strip().split(','))
        edges.append((id_mapping[id1], id_mapping[id2]))
        
        sparse_matrix[id_mapping[id1], id_mapping[id2]] = 1
        sparse_matrix[id_mapping[id2], id_mapping[id1]] = 1
        
print('Loaded matrix, starting finding eigenvector')

A = sparse_matrix.tocsr()

def tpower(A: csr_matrix, K: int, max_iters=5000, tol=1e-6):
    n = A.shape[0]
    # init: top-eigenvector, made nonnegative
    _, v = eigsh(A, k=1, which='LA', tol=5e-7, maxiter=1000)
    z = np.abs(v.ravel())
    
    #find initial density
    idxes = np.argpartition(np.abs(z), -K)[-K:]
    X_i = np.zeros(n, bool)
    X_i[idxes] = True
    density = (X_i.T @ A @ X_i) / (K * (K - 1))
    print('Initial Density: ', density)
    
    z /= np.linalg.norm(z)

    for it in range(max_iters):
        y = A.dot(z)
        # truncate: keep only k largest entries
        idx = np.argpartition(y, -K)[-K:]
        y_trunc = np.zeros_like(y)
        y_trunc[idx] = y[idx]
        norm = np.linalg.norm(y_trunc)
        if norm == 0:
            break
        z_new = y_trunc / norm

        # check convergence
        if np.linalg.norm(z_new - z) < tol:
            break
        z = z_new

    # final rounding
    final_idx = np.argpartition(np.abs(z), -K)[-K:]
    X = np.zeros(n, bool)
    X[final_idx] = True
    return X, z

X_t, z_t = tpower(A, K)

print('Non-zero X_t', (X_t != 0).sum())
print('Final Density: ', (X_t.T @ A @ X_t) / (K * (K - 1)))

# get the actual elements of the top K
topK_idx = np.argpartition(np.abs(z_t), -K)[-K:]
print('Top K elements:\n', topK_idx.tolist())