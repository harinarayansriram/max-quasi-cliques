import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def projected_gradient_ascent(A, K, eta=0.0005, max_iters=100_000, tol=1e-6):
    '''
    A:      csr_matrix or array, your adjacency (n x n)
    K:      target sparsity (number of nonzeros in z)
    eta:    step-size for the gradient ascent
    max_iters: maximum iterations
    tol:    convergence tolerance on z-changes
    '''
    n = A.shape[0]
    # use the top eigenvector as a warm start
    _, v = eigsh(A, k=1, which='LA', tol=1e-6, maxiter=1000)
    z = v.ravel()
    z /= np.linalg.norm(z)


    for _ in range(max_iters):
        # gradient step: ascend along grad_z (z^T A z)' = 2 A z (scalar 2 absorbed in eta)
        y = z + eta * (A.dot(z))

        # projection (l0‐truncate): keep only the K largest‐magnitude entries
        idx = np.argpartition(np.abs(y), -K)[-K:]
        z_trunc = np.zeros_like(y)
        z_trunc[idx] = y[idx]

        # renormalize
        norm = np.linalg.norm(z_trunc)
        if norm < tol:
            break
        z_new = z_trunc / norm

        if np.linalg.norm(z_new - z) < tol:
            z = z_new
            break
        z = z_new

    # final rounding to get the discrete X
    final_idx = np.argpartition(np.abs(z), -K)[-K:]
    X = np.zeros(n, dtype=bool)
    X[final_idx] = True

    return X, z


def binary_search_in_range(A, threshold, low, high):
    best_k = None
    best_result = None
    
    while low <= high:
        mid = (low + high) // 2
        X_mid, z_mid = projected_gradient_ascent(A, mid)
        mid_density = (X_mid.T @ A @ X_mid) / (mid * (mid - 1))
        print(f'  Binary search: K={mid}, density={mid_density:.6f}')
        
        if mid_density >= threshold:
            best_k = mid
            best_result = (mid_density, X_mid, z_mid)
            low = mid + 1
        else:
            high = mid - 1
    
    if best_k is not None:
        return best_k, best_result[0], best_result[1], best_result[2] # (optimal_k, density, X, z)
    return None

adaptive_step_cap = 512
def find_optimal_k_adaptive(A, threshold, initial_k, max_k=None):
    n = A.shape[0]
    if max_k is None:
        max_k = n - 1
    
    print(f'Starting adaptive search for optimal K with threshold {threshold}')
    
    # test initial k
    X_curr, z_curr = projected_gradient_ascent(A, initial_k)
    curr_density = (X_curr.T @ A @ X_curr) / (initial_k * (initial_k - 1))
    print(f'Initial K={initial_k}, density={curr_density:.6f}')
    
    if curr_density >= threshold:
        current_k = initial_k
        step_size = 1
        last_good_k = current_k
        
        while current_k + step_size <= max_k:
            test_k = current_k + step_size
            X_test, z_test = projected_gradient_ascent(A, test_k)
            test_density = (X_test.T @ A @ X_test) / (test_k * (test_k - 1))
            print(f'Exponential search: K={test_k}, density={test_density:.6f}, step={step_size}')
            
            if test_density >= threshold:
                last_good_k = test_k
                current_k = test_k
                step_size = min(step_size * 2, adaptive_step_cap)  # exponential growth, capped
            else:
                # went too far, binary search between last_good_k and test_k
                print(f'Switching to binary search between K={last_good_k} and K={test_k}')
                result = binary_search_in_range(A, threshold, last_good_k + 1, test_k - 1)
                if result is not None:
                    final_k, final_density, final_X, final_z = result
                    # check if we found something better than last_good_k
                    if final_k > last_good_k:
                        last_good_k = final_k
                break
        
        # get final result for last_good_k
        X_final, z_final = projected_gradient_ascent(A, last_good_k)
        final_density = (X_final.T @ A @ X_final) / (last_good_k * (last_good_k - 1))
        
    else:
        # density too low, use exponential steps downward then binary search
        current_k = initial_k
        step_size = 1
        lower_bound = 2 
        
        while current_k - step_size >= lower_bound:
            test_k = current_k - step_size
            X_test, z_test = projected_gradient_ascent(A, test_k)
            test_density = (X_test.T @ A @ X_test) / (test_k * (test_k - 1))
            print(f'Exponential search down: K={test_k}, density={test_density:.6f}, step={step_size}')
            
            if test_density >= threshold:
                # found a good region, binary search between test_k and current_k
                print(f'Switching to binary search between K={test_k} and K={current_k}')
                result = binary_search_in_range(A, threshold, test_k, current_k - 1)
                if result is not None:
                    last_good_k, final_density, X_final, z_final = result
                    break
            
                last_good_k = test_k
                final_density = test_density
                X_final = X_test
                z_final = z_test
                break
            else:
                current_k = test_k
                step_size = min(step_size * 2, adaptive_step_cap)
        else:
            # exhausted downward search, do final binary search
            print(f'Final binary search between K={lower_bound} and K={current_k}')
            result = binary_search_in_range(A, threshold, lower_bound, current_k - 1)
            if result is not None:
                last_good_k, final_density, X_final, z_final = result
            else:
                raise ValueError(f'No K >= {lower_bound} satisfies density threshold {threshold}')
    
    print(f'\nOptimal K found: {last_good_k}')
    print(f'Final density: {final_density:.6f}')
    print(f'Threshold: {threshold}')
    
    return last_good_k, final_density, X_final, z_final

if __name__ == '__main__':
    id_mapping = {}
    with open('id_mapping.csv', 'r') as f:
        for line in f:
            id1, idx = map(int, line.strip().split(' '))
            id_mapping[id1] = idx
            
    rev_mapping = {v: k for k, v in id_mapping.items()}
    
    sparse_matrix = sp.lil_matrix((len(id_mapping), len(id_mapping)))
    with open('edges.csv', 'r') as f:
        for line in f:
            id1, id2 = map(int, line.strip().split(','))
            sparse_matrix[id_mapping[id1], id_mapping[id2]] = 1
            sparse_matrix[id_mapping[id2], id_mapping[id1]] = 1
    
    A = sparse_matrix.tocsr()
    print('Loaded matrix, starting optimization')
    
    denoms = [2**k for k in range(1,12)]
    # denoms = [8,16]
    init_ks = [127, 298, 623, 1447, 2722, 5690, 10704, 20139, 34106, 57760, 57760]
    # init_ks = init_ks[2:4]
    summary_stats = []
    
    for denom, init_k in zip(denoms, init_ks):  
        print('\n\n\n','-'*50)
        threshold = 1.0/denom
        initial_k = init_k
        start_time = time.time()
        
        print(f'Running with threshold: {threshold} (denom: {denom}), initial K: {initial_k}')
        optimal_k, final_density, X_final, z_final = find_optimal_k_adaptive(
            A, threshold, initial_k
        )
        
        end_time = time.time()
        
        print(f'Time taken: {end_time - start_time:.2f} seconds')
        print(f'\nFinal Results:')
        print(f'Optimal K: {optimal_k}')
        print(f'Final Density: {final_density:.6f}')
        print(f'Non-zero X_final: {(X_final != 0).sum()}')
        
        summary_stats.append((denom, initial_k, optimal_k, final_density, end_time - start_time))
        
        # get the actual top k indices
        topK_idx = np.argpartition(np.abs(z_final), -optimal_k)[-optimal_k:]
        # print(f'Top {optimal_k} elements: {topK_idx.tolist()}')
        topK_elements = [rev_mapping[idx] for idx in topK_idx]
        
        with open(f'projected_spectral_results/run_3/{denom}_result.csv', 'w') as f:
            f.write('\n'.join(map(str, topK_elements)))
            
    with open('projected_spectral_results/run_3/summary_stats.csv', 'w') as f:
        f.write('denom,initial_k,optimal_k,final_density,time_taken')
        for denom, init_k, optimal_k, final_density, time_taken in summary_stats:
            f.write(f'\n{denom},{init_k},{optimal_k},{final_density},{time_taken}')