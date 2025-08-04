import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def tpower(A: csr_matrix, K: int, max_iters=5000, tol=1e-6):
    n = A.shape[0]
    # 1) init: top-eigenvector, made nonnegative
    _, v = eigsh(A, k=1, which='LA', tol=5e-7, maxiter=1000)
    # z = np.abs(v.ravel())
    z = v.ravel()
    
    #find initial density
    # idxes = np.argpartition(np.abs(z), -K)[-K:]
    # X_i = np.zeros(n, bool)
    # X_i[idxes] = True
    # density = (X_i.T @ A @ X_i) / (K * (K - 1))
    
    z /= np.linalg.norm(z)

    alpha = 0.8
    for it in range(max_iters):
        y = A.dot(z)
        # truncate: keep only K largest entries
        abs_y = np.abs(y)
        # idx = np.argpartition(y, -K)[-K:]
        idx = np.argpartition(abs_y, -K)[-K:]
        y_trunc = np.zeros_like(y)
        # y_trunc[idx] = y[idx]
        y_trunc[idx] = np.sign(y[idx]) * abs_y[idx]
        norm = np.linalg.norm(y_trunc)
        if norm == 0:
            break
        
        y_damped = alpha * y_trunc + (1 - alpha) * z
        z_new = y_damped / np.linalg.norm(y_damped)

        # check convergence
        if np.linalg.norm(z_new - z) < tol:
            break
        z = z_new

    # final rounding
    final_idx = np.argpartition(np.abs(z), -K)[-K:]
    X = np.zeros(n, bool)
    X[final_idx] = True
    return X, z

def find_optimal_k_binary_search(A, threshold, initial_k, max_k=None):
    '''
    Find the largest K such that density >= threshold using binary search.
    
    Args:
        A: adjacency matrix
        threshold: minimum density threshold
        initial_k: starting guess for K
        max_k: maximum K to consider (default: number of nodes)
    
    Returns:
        tuple: (optimal_k, final_density, X, z)
    '''
    n = A.shape[0]
    if max_k is None:
        max_k = n
    
    print(f'Starting binary search for optimal K with threshold {threshold}')
    
    # First, check if initial_k gives us density >= threshold
    X_init, z_init = tpower(A, initial_k)
    init_density = (X_init.T @ A @ X_init) / (initial_k * (initial_k - 1))
    print(f'Initial K={initial_k}, density={init_density:.6f}')
    
    if init_density < threshold:
        # Need to search downward
        high = initial_k
        low = 2  # minimum meaningful K
        
        # Find a lower bound where density >= threshold
        while low < high:
            mid = (low + high) // 2
            X_mid, z_mid = tpower(A, mid)
            mid_density = (X_mid.T @ A @ X_mid) / (mid * (mid - 1))
            print(f'Testing K={mid}, density={mid_density:.6f}')
            
            if mid_density >= threshold:
                low = mid + 1  # Try larger K
            else:
                high = mid     # Current K too large
        
        # low-1 is our answer
        optimal_k = low - 1
        
    else:
        # Density is good, try to find larger K
        low = initial_k
        high = max_k
        
        # Find upper bound where density < threshold
        while low < high:
            mid = (low + high + 1) // 2  # Bias toward upper half
            if mid >= n:  # Can't have K >= n
                high = n - 1
                continue
                
            X_mid, z_mid = tpower(A, mid)
            mid_density = (X_mid.T @ A @ X_mid) / (mid * (mid - 1))
            print(f'Testing K={mid}, density={mid_density:.6f}')
            
            if mid_density >= threshold:
                low = mid      # Can go larger
            else:
                high = mid - 1 # Current K too large
        
        optimal_k = low
    
    # Get final result
    X_final, z_final = tpower(A, optimal_k)
    final_density = (X_final.T @ A @ X_final) / (optimal_k * (optimal_k - 1))
    
    print(f'\nOptimal K found: {optimal_k}')
    print(f'Final density: {final_density:.6f}')
    print(f'Threshold: {threshold}')
    print(f'Meets threshold: {final_density >= threshold}')
    
    return optimal_k, final_density, X_final, z_final

def binary_search_in_range(A, threshold, low, high):
    '''
    Helper function to do binary search between low and high (inclusive).
    Returns (optimal_k, density, X, z) or None if no valid K found.
    '''
    best_k = None
    best_result = None
    
    while low <= high:
        mid = (low + high) // 2
        X_mid, z_mid = tpower(A, mid)
        mid_density = (X_mid.T @ A @ X_mid) / (mid * (mid - 1))
        print(f'  Binary search: K={mid}, density={mid_density:.6f}')
        
        if mid_density >= threshold:
            # This K works, save it and try larger
            best_k = mid
            best_result = (mid_density, X_mid, z_mid)
            low = mid + 1
        else:
            # This K doesn't work, try smaller
            high = mid - 1
    
    if best_k is not None:
        return best_k, best_result[0], best_result[1], best_result[2]
    return None



adaptive_step_cap = 512
def find_optimal_k_adaptive(A, threshold, initial_k, max_k=None):
    '''
    Alternative approach: adaptive step search with exponential scaling.
    Faster for cases where optimal K is much larger than initial guess.
    Uses binary search for final refinement.
    '''
    n = A.shape[0]
    if max_k is None:
        max_k = n - 1
    
    print(f'Starting adaptive search for optimal K with threshold {threshold}')
    
    # Test initial K
    X_curr, z_curr = tpower(A, initial_k)
    curr_density = (X_curr.T @ A @ X_curr) / (initial_k * (initial_k - 1))
    print(f'Initial K={initial_k}, density={curr_density:.6f}')
    
    if curr_density >= threshold:
        # Density is good, try larger K with exponential steps
        current_k = initial_k
        step_size = 1
        last_good_k = current_k
        
        while current_k + step_size <= max_k:
            test_k = current_k + step_size
            X_test, z_test = tpower(A, test_k)
            test_density = (X_test.T @ A @ X_test) / (test_k * (test_k - 1))
            print(f'Exponential search: K={test_k}, density={test_density:.6f}, step={step_size}')
            
            if test_density >= threshold:
                # Still good, continue with larger steps
                last_good_k = test_k
                current_k = test_k
                step_size = min(step_size * 2, adaptive_step_cap)  # Exponential growth, capped
            else:
                # Went too far, binary search between last_good_k and test_k
                print(f'Switching to binary search between K={last_good_k} and K={test_k}')
                result = binary_search_in_range(A, threshold, last_good_k + 1, test_k - 1)
                if result is not None:
                    final_k, final_density, final_X, final_z = result
                    # Check if we found something better than last_good_k
                    if final_k > last_good_k:
                        last_good_k = final_k
                break
        
        # Get final result for last_good_k
        X_final, z_final = tpower(A, last_good_k)
        final_density = (X_final.T @ A @ X_final) / (last_good_k * (last_good_k - 1))
        
    else:
        # Density too low, use exponential steps downward then binary search
        current_k = initial_k
        step_size = 1
        lower_bound = 2  # minimum meaningful K
        
        # Find a rough lower bound with exponential steps
        while current_k - step_size >= lower_bound:
            test_k = current_k - step_size
            X_test, z_test = tpower(A, test_k)
            test_density = (X_test.T @ A @ X_test) / (test_k * (test_k - 1))
            print(f'Exponential search down: K={test_k}, density={test_density:.6f}, step={step_size}')
            
            if test_density >= threshold:
                # Found a good region, binary search between test_k and current_k
                print(f'Switching to binary search between K={test_k} and K={current_k}')
                result = binary_search_in_range(A, threshold, test_k, current_k - 1)
                if result is not None:
                    last_good_k, final_density, X_final, z_final = result
                else:
                    # Use test_k as fallback
                    last_good_k = test_k
                    final_density = test_density
                    X_final = X_test
                    z_final = z_test
                break
            else:
                current_k = test_k
                step_size = min(step_size * 2, adaptive_step_cap)
        else:
            # Exhausted downward search, do final binary search
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

# Example usage with your existing code:
if __name__ == '__main__':
    # Your existing loading code
    
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
    init_ks = init_ks[2:4]
    summary_stats = []
    
    for denom, init_k in zip(denoms, init_ks):
        print('\n\n\n','-'*50)
        threshold = 1.0/denom
        initial_k = init_k
        start_time = time.time()
        
        print(f'Running with threshold: {threshold}, initial K: {initial_k}')
        # Method 2: Adaptive search (potentially faster for large jumps)
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
        
        # Get the actual top K indices
        topK_idx = np.argpartition(np.abs(z_final), -optimal_k)[-optimal_k:]
        # print(f'Top {optimal_k} elements: {topK_idx.tolist()}')
        topK_elements = [rev_mapping[idx] for idx in topK_idx]
        
        # with open(f'spectral_results/{denom}_result.csv', 'w') as f:
        #     f.write('\n'.join(map(str, topK_elements)))
            
    # with open('spectral_results/summary_stats.csv', 'w') as f:
    #     f.write('denom,initial_k,optimal_k,final_density,time_taken')
    #     f.write('\n'.join(map(','.join, summary_stats)))
        
    